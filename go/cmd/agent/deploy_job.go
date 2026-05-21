package main

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"sort"
	"strings"

	"github.com/Jorrit05/DYNAMOS/pkg/etcd"
	"github.com/Jorrit05/DYNAMOS/pkg/mschain"
	pb "github.com/Jorrit05/DYNAMOS/pkg/proto"
	"go.opencensus.io/trace"
	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Necessary for port selection. Original DYNAMOS just does last port +1 i think, but this caused issues for me so i hardcoded it
func sortMicroserviceChain(msChain []mschain.MicroserviceMetadata) {
	priority := map[string]int{
		"anonymize-test":        0,
		"differential-p-test":   1,
		"vfl-train-model-demo":  2,
		"vfl-train-demo":        3,
	}

	sort.SliceStable(msChain, func(i, j int) bool {
		a := strings.ToLower(msChain[i].Name)
		b := strings.ToLower(msChain[j].Name)

		pi, okI := priority[a]
		pj, okJ := priority[b]

		if okI && okJ {
			return pi < pj
		}
		if okI != okJ {
			return okI
		}
		return a < b
	})
}

func jobExists(ctx context.Context, jobName string) bool {
	// Check if job exists, if it does, do not make a new one
	dataStewardName := strings.ToLower(serviceName)
	if dataStewardName == "" {
		return false
	}

	jobMutex.Lock()
	newValue := jobCounter[jobName]
	jobMutex.Unlock()

	newJobName := replaceLastCharacter(jobName, newValue)

	logger.Sugar().Info("Steward: ", dataStewardName, ", job name: ", newJobName)
	logger.Sugar().Info(clientSet.BatchV1().Jobs(dataStewardName))
	_, err := clientSet.BatchV1().Jobs(dataStewardName).Get(ctx, newJobName, metav1.GetOptions{})

	return err == nil
}

func generateChainAndDeploy(ctx context.Context, compositionRequest *pb.CompositionRequest, localJobName string, options map[string]bool) (context.Context, *batchv1.Job, error) {
	logger.Debug("Starting generateChainAndDeploy")

	ctx, span := trace.StartSpan(ctx, serviceName+"/func: generateChainAndDeploy")
	defer span.End()

	msChain, err := generateMicroserviceChain(compositionRequest, options)
	if err != nil {
		logger.Sugar().Errorf("Error generating microservice chain %v", err)
		return ctx, nil, err
	}
	logger.Sugar().Debug(msChain)

	createdJob, err := deployJob(ctx, msChain, localJobName, compositionRequest)
	if err != nil {
		logger.Sugar().Errorf("Error deploying job %v", err)
		return ctx, nil, err
	}

	return ctx, createdJob, nil
}

func deployJob(ctx context.Context, msChain []mschain.MicroserviceMetadata, jobName string, compositionRequest *pb.CompositionRequest) (*batchv1.Job, error) {
	logger.Debug("Starting deployJob")

	dataStewardName := strings.ToLower(serviceName)
	if dataStewardName == "" {
		return nil, fmt.Errorf("env variable DATA_STEWARD_NAME not defined")
	}

	jobMutex.Lock()
	jobCounter[jobName]++
	newValue := jobCounter[jobName]
	jobMutex.Unlock()

	newJobName := replaceLastCharacter(jobName, newValue)

	// --- IMPORTANT: build PodSpec outside the Job struct ---
	podSpec := v1.PodSpec{
		Containers:    []v1.Container{},
		RestartPolicy: v1.RestartPolicyOnFailure,
		Volumes: []v1.Volume{
			{
				Name: "shared-data",
				VolumeSource: v1.VolumeSource{
					EmptyDir: &v1.EmptyDirVolumeSource{},
				},
			},
		},
	}

	// In local dev (single-node docker-desktop), do NOT pin NodeName to "clientone"/"server"/etc.
	// Otherwise pods will sit Pending forever and then get killed by deadlines/backoff.
	if strings.ToLower(os.Getenv("LOCAL_DEV")) != "true" {
		podSpec.NodeName = dataStewardName
	}

	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      newJobName,
			Namespace: dataStewardName,
			Labels:    map[string]string{"app": dataStewardName, "jobName": jobName},
		},
		Spec: batchv1.JobSpec{
			ActiveDeadlineSeconds:   &activeDeadlineSeconds,
			TTLSecondsAfterFinished: &ttl, // Clean up job TTL after it finishes
			BackoffLimit:            &backoffLimit,
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"app": dataStewardName, "nodeName": dataStewardName},
				},
				Spec: podSpec,
			},
		},
	}

	// Add the containers to the job
	port := firstPortMicroservice
	nrOfServices := len(msChain)
	firstService := "1"
	lastService := "0"

	// Determine the amount of data providers
	nrOfDataProviders := 0
	if compositionRequest.DataProviders != nil {
		nrOfDataProviders = len(compositionRequest.DataProviders)
	}

	// As mentioned before, first sort then loop over sorted chain
	sortMicroserviceChain(msChain)
	logger.Sugar().Infow("Sorted chain order", "chain", func() []string {
		out := make([]string, 0, len(msChain))
		for _, ms := range msChain {
			out = append(out, ms.Name)
		}
		return out
	}())

	for i, microservice := range msChain {
		port++

		if i == nrOfServices-1 {
			lastService = "1"
		}

		logger.Sugar().Debugw("job info:", "name: ", microservice.Name, "Port: ", port)

		microserviceTag := getMicroserviceTag(microservice.Name)

		repositoryName := os.Getenv("MICROSERVICE_REPOSITORY_NAME")
		if repositoryName == "" {
			repositoryName = "tamjiyan" // you changed this
		}

		fullImage := fmt.Sprintf("%s/%s:%s", repositoryName, microservice.Name, microserviceTag)
		logger.Sugar().Debugf("FullImage name: %s", fullImage)
		logger.Sugar().Debugf("JobName: %s", jobName)

		container := v1.Container{
			Name:            microservice.Name,
			Image:           fullImage,
			ImagePullPolicy: v1.PullAlways,
			Env: []v1.EnvVar{
				{Name: "DATA_STEWARD_NAME", Value: strings.ToUpper(dataStewardName)},
				{Name: "DESIGNATED_GRPC_PORT", Value: strconv.Itoa(port)},
				{Name: "FIRST", Value: firstService},
				{Name: "LAST", Value: lastService},
				{Name: "JOB_NAME", Value: jobName},
				{Name: "SIDECAR_PORT", Value: strconv.Itoa(firstPortMicroservice - 1)},
				{Name: "OC_AGENT_HOST", Value: tracingHost},
				{Name: "NR_OF_DATA_PROVIDERS", Value: strconv.Itoa(nrOfDataProviders)},
				{Name: "ANON_DIR", Value: "/shared"},
			},
			VolumeMounts: []v1.VolumeMount{
				{
					Name:      "shared-data",
            		MountPath: "/shared",
				},
			},
		}

		job.Spec.Template.Spec.Containers = append(job.Spec.Template.Spec.Containers, container)
		firstService = "0"
	}

	job.Spec.Template.Spec.Containers = append(job.Spec.Template.Spec.Containers, addSidecar())

	if clientSet == nil {
		clientSet = getKubeClient()
	}

	createdJob, err := clientSet.BatchV1().Jobs(dataStewardName).Create(ctx, job, metav1.CreateOptions{})
	if err != nil {
		logger.Sugar().Errorf("failed to create job: %v", err)
		return nil, err
	}

	return createdJob, nil
}

func addSidecar() v1.Container {
	sidecarName := os.Getenv("SIDECAR_NAME")
	if sidecarName == "" {
		sidecarName = "sidecar"
	}

	repositoryName := os.Getenv("SIDECAR_REPOSITORY_NAME")
	if repositoryName == "" {
		repositoryName = "tamjiyan"
	}

	sidecarTag := getMicroserviceTag(sidecarName)

	fullImage := fmt.Sprintf("%s/%s:%s", repositoryName, sidecarName, sidecarTag)
	logger.Sugar().Debugf("Sidecar name: %s", fullImage)

	return v1.Container{
		Name:            sidecarName,
		Image:           fullImage,
		ImagePullPolicy: v1.PullAlways,
		Env: []v1.EnvVar{
			{Name: "DESIGNATED_GRPC_PORT", Value: strconv.Itoa(firstPortMicroservice - 1)},
			{Name: "TEMPORARY_JOB", Value: "true"},
			{Name: "AMQ_USER", Value: rabbitMqUser},
			{Name: "OC_AGENT_HOST", Value: tracingHost},
			{
				Name: "AMQ_PASSWORD",
				ValueFrom: &v1.EnvVarSource{
					SecretKeyRef: &v1.SecretKeySelector{
						LocalObjectReference: v1.LocalObjectReference{Name: "rabbit"},
						Key:                  "password",
					},
				},
			},
		},
	}
}

func getRequiredMicroservices(microserviceMetada *[]mschain.MicroserviceMetadata, request *mschain.RequestType, role string) error {
	logger.Sugar().Debug("started getRequiredMicroservices")
	for _, ms := range request.RequiredServices {
		var metadataObject mschain.MicroserviceMetadata

		_, err := etcd.GetAndUnmarshalJSON(etcdClient, fmt.Sprintf("/microservices/%s/chainMetadata", ms), &metadataObject)
		if err != nil {
			return err
		}

		if strings.EqualFold(metadataObject.Label, role) {
			*microserviceMetada = append(*microserviceMetada, metadataObject)
		} else if strings.EqualFold("all", role) {
			*microserviceMetada = append(*microserviceMetada, metadataObject)
		}
	}

	return nil
}

func getOptionalMicroservices(microserviceMetada *[]mschain.MicroserviceMetadata, request *mschain.RequestType, role string, requestType string, options map[string]bool) error {
	logger.Debug("Start getOptionalMicroservices")
	logger.Sugar().Debug(len(request.OptionalServices))
	// Again made changes for sorting. Now always processes optional services in same order
	optionKeys := make([]string, 0, len(options))
	for k := range options {
		optionKeys = append(optionKeys, k)
	}
	sort.Strings(optionKeys)

	msNames := make([]string, 0, len(request.OptionalServices))
	for msName := range request.OptionalServices {
		msNames = append(msNames, msName)
	}
	sort.Strings(msNames)

	for _, option := range optionKeys {
		boolVal := options[option]
		logger.Sugar().Debugf("Option: %s boolVal: %b", option, boolVal)

		if !boolVal {
			continue
		}

		for _, msName := range msNames {
			optionKey := request.OptionalServices[msName]

			if strings.EqualFold(option, optionKey) {
				var metadataObject mschain.MicroserviceMetadata

				_, err := etcd.GetAndUnmarshalJSON(
					etcdClient,
					fmt.Sprintf("/microservices/%s/chainMetadata", msName),
					&metadataObject,
				)
				if err != nil {
					return err
				}

				if strings.EqualFold(metadataObject.Label, role) || strings.EqualFold("all", role) {
					*microserviceMetada = append(*microserviceMetada, metadataObject)
				}
			}
		}
	}

	return nil
}

func RequestTypeMicroservices(requestType string) (mschain.RequestType, error) {
	var request mschain.RequestType
	_, err := etcd.GetAndUnmarshalJSON(etcdClient, fmt.Sprintf("/requestTypes/%s", requestType), &request)
	if err != nil {
		return mschain.RequestType{}, err
	}

	return request, nil
}

func replaceLastCharacter(name string, replaceWith int) string {
	if name == "" {
		return ""
	}

	nameSlice := []rune(name)
	nameSlice = nameSlice[:len(nameSlice)-1]

	runeSlice := []rune(strconv.Itoa(replaceWith))
	nameSlice = append(nameSlice, runeSlice...)

	return string(nameSlice)
}

func getMicroserviceTag(msName string) string {
	tag := os.Getenv(fmt.Sprintf("%s_TAG", strings.ToUpper(msName)))
	if tag != "" {
		return tag
	}
	return "latest"
}
