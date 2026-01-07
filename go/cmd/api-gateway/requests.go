// This file contains the handlers for the requests that the API Gateway receives from the client
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"
	"github.com/Jorrit05/DYNAMOS/pkg/api"
	"github.com/Jorrit05/DYNAMOS/pkg/lib"
	pb "github.com/Jorrit05/DYNAMOS/pkg/proto"
	"github.com/google/uuid"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.opencensus.io/trace"
)

const (
    StatusPending = "pending"
    StatusDone    = "done"
    StatusFailed  = "failed"
)

var (
	activeJobID      string
	activeJobLock    sync.Mutex   // to allow only 1 active job at any time
	trainingRequests = sync.Map{} // map[string]TrainingRequestData
)

type TrainingRequestData struct {
	Status   string
	Results  []map[string]any
	Metadata map[string]any
	// add more fields as needed
}


func getTrainingStatusHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		logger.Sugar().Info("Starting getTrainingStatusHandler")
		requestID := r.URL.Query().Get("id")
		v, ok := trainingRequests.Load(requestID)
		if !ok {
			http.Error(w, "Request ID not found", http.StatusNotFound)
			return
		}

		logger.Sugar().Debug("Found training request: ", requestID)
		reqData := v.(TrainingRequestData)
		resp := map[string]any{
			"request_id": requestID,
			"status":     reqData.Status,
			"metadata":   reqData.Metadata,
			"results":    reqData.Results,
		}
		respBytes, _ := json.MarshalIndent(resp, "", "    ")
		w.WriteHeader(http.StatusOK)
		w.Write(respBytes)
	}
}

func requestHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		activeJobLock.Lock()
		defer activeJobLock.Unlock()

		// Check for existing active job
		if activeJobID != "" {
			v, _ := trainingRequests.Load(activeJobID)
			reqData := v.(TrainingRequestData)
			resp := map[string]any{
				"error":             "A training job is already in progress.",
				"active_request_id": activeJobID,
				"active_status":     reqData.Status,
			}
			respBytes, _ := json.Marshal(resp)
			w.WriteHeader(http.StatusTooManyRequests)
			w.Write(respBytes)
			return
		}

		// Accept new job
		requestID := uuid.New().String()
		activeJobID = requestID
		reqData := TrainingRequestData{
			Status: StatusPending,
			Metadata: map[string]any{
				"created_at": time.Now().Format(time.RFC3339),
			},
			Results: []map[string]any{},
		}
		trainingRequests.Store(requestID, reqData)

		resp := map[string]any{
			"request_id": requestID,
			"status":     StatusPending,
		}

		logger.Sugar().Info("Accepted new job with id: ", activeJobID)

		// Parse the request body
		body, err := api.GetRequestBody(w, r, serviceName)
		if err != nil {
			return
		}

		var apiReqApproval api.RequestApproval
		if err := json.Unmarshal(body, &apiReqApproval); err != nil {
			logger.Sugar().Errorf("Error unmMarshalling get apiReqApproval: %v", err)
			return
		}

		userPb := &pb.User{
			Id:       apiReqApproval.User.Id,
			UserName: apiReqApproval.User.UserName,
		}

		var dataRequestInterface map[string]any
		if err := json.Unmarshal(apiReqApproval.DataRequest, &dataRequestInterface); err != nil {
			logger.Sugar().Errorf("Error unmarhsalling get request: %v", err)
			return
		}

		dataRequestOptions := &api.DataRequestOptions{}
		dataRequestOptions.Options = make(map[string]bool)
		if err := json.Unmarshal(apiReqApproval.DataRequest, &dataRequestOptions); err != nil {
			logger.Sugar().Errorf("Error unmMarshalling get apiReqApproval: %v", err)
			return
		}

		dataRequestInterface["user"] = userPb

		// Create protobuf struct for the req approval flow
		protoRequest := &pb.RequestApproval{
			Type:             apiReqApproval.Type,
			User:             userPb,
			DataProviders:    apiReqApproval.DataProviders,
			DestinationQueue: "policyEnforcer-in",
			Options:          dataRequestOptions.Options,
		}

		respBytes, _ := json.Marshal(resp)
		w.WriteHeader(http.StatusAccepted)
		w.Write(respBytes)

		// ---- TRIGGER TRAINING IN BACKGROUND ----
		go func() {
			startTraining(protoRequest, dataRequestInterface, apiReqApproval, r, requestID)
		}()

	}
}

func startTraining(protoRequest *pb.RequestApproval, dataRequestInterface map[string]any, apiReqApproval api.RequestApproval, r *http.Request, requestID string) {
	logger.Debug("Starting training process...")
	// Requests may take up to 10 minutes now
	ctxWithTimeout, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// Start a new span with the context that has a timeout
	ctx, span := trace.StartSpan(ctxWithTimeout, "requestApprovalHandler")
	defer span.End()

	// Create a channel to receive the response
	responseChan := make(chan validation)

	requestApprovalMutex.Lock()
	requestApprovalMap[protoRequest.User.Id] = responseChan
	requestApprovalMutex.Unlock()

	_, err := c.SendRequestApproval(ctx, protoRequest)
	if err != nil {
		logger.Sugar().Errorf("error in sending requestapproval: %v", err)
	}

	select {
	case validationStruct := <-responseChan:
		msg := validationStruct.response

		logger.Sugar().Infof("Received response, %s", msg.Type)
		if msg.Type != "requestApprovalResponse" {
			logger.Sugar().Errorf("Unexpected message received, type: %s", msg.Type)
			// http.Error(w, "Internal server error", http.StatusInternalServerError)
			return
		}

		requestMetadata := &pb.RequestMetadata{
			JobId: msg.JobId,
		}
		dataRequestInterface["requestMetadata"] = requestMetadata

		logger.Sugar().Infof("Data Prepared jsonData: %s", dataRequestInterface)

		var response []byte

		if apiReqApproval.Type == "vflTrainModelRequest" {
			ctxWithoutCancel := context.WithoutCancel(r.Context())
			response = runVFLTraining(dataRequestInterface, msg.AuthorizedProviders, msg.JobId, ctxWithoutCancel, requestID)

		} else {
			// Marshal the combined data back into JSON for forwarding
			dataRequestJson, err := json.Marshal(dataRequestInterface)
			if err != nil {
				logger.Sugar().Errorf("Error marshalling combined data: %v", err)
				return
			}

			response = sendDataToAuthProviders(dataRequestJson, msg.AuthorizedProviders, apiReqApproval.Type, msg.JobId)
		}

		// w.WriteHeader(http.StatusOK)
		// w.Write(response)
		logger.Sugar().Info("Training process completed for request id: ", requestID)
		logger.Sugar().Info("Response: ", string(response))
		return

	case <-ctx.Done():
		// http.Error(w, "Request timed out", http.StatusRequestTimeout)
		return
	}

}

func runVFLTrainingRound(dataRequest map[string]any, clients map[string]string, serverAuth string, serverUrl string, learning_rate float64, trainingBacktrack int64) (float64, error) {
	var wg sync.WaitGroup
	responses := map[string]string{}

	for auth, url := range clients {

		logger.Sugar().Info("Sending training request to client: ", auth, " at url: ", url)

		wg.Add(1)
		target := strings.ToLower(auth)

		endpoint := fmt.Sprintf("http://%s:8080/agent/v1/vflTrainRequest/%s", url, target)

		dataRequest["type"] = "vflTrainRequest"

		dataRequestJson, err := json.Marshal(dataRequest)
		if err != nil {
			logger.Sugar().Errorf("Error marshalling combined data: %v", err)
			return 0., err
		}

		go func() {
			responseData, err := sendData(endpoint, dataRequestJson)

			if err != nil {
				logger.Sugar().Errorf("Error sending data, %v", err)
			} else {
				responseJson := &pb.MicroserviceCommunication{}
				err = json.Unmarshal([]byte(responseData), responseJson)

				if err != nil {
					logger.Sugar().Error("Unmarshalling response did not go well: ", err)
				}

				dataJson := responseJson.Data.AsMap()
				embeddings, ok := dataJson["embeddings"].(string)

				if !ok {
					logger.Sugar().Error("No embeddings found in the return data.")
					embeddings = ""
					// TODO: Handle disagreements?
				}

				responses[target] = embeddings
			}

			wg.Done()
		}()
	}

	wg.Wait()

	target := strings.ToLower(serverAuth)
	logger.Sugar().Info("Sending training request to server: ", target, " at url: ", serverUrl)

	endpoint := fmt.Sprintf("http://%s:8080/agent/v1/vflTrainRequest/%s", serverUrl, target)

	dataRequest["type"] = "vflAggregateRequest"

	// note: changed this to be dynamic based on the clients available
	// dataRequest["data"] = map[string]any{
	// 	"embeddings": []string{responses["clientone"], responses["clienttwo"], responses["clientthree"]},
	// }
	// Collect embeddings from all clients
	embeddingList := []string{}
	for approved_client := range clients {
		if emb, ok := responses[strings.ToLower(approved_client)]; ok {
			embeddingList = append(embeddingList, emb)
		}
	}

	// Prepare data for aggregation
	dataRequest["type"] = "vflAggregateRequest"
	dataRequest["data"] = map[string]any{
		"embeddings":        embeddingList,
		"trainingBacktrack": trainingBacktrack,
	}

	dataRequestJson, err := json.Marshal(dataRequest)
	if err != nil {
		logger.Sugar().Errorf("Error marshalling combined data: %v", err)
		return 0., err
	}

	responseData, error := sendData(endpoint, dataRequestJson)
	if error != nil {
		logger.Sugar().Errorf("Error sending data to the server, %v", error)
	}

	serverResponse := &pb.MicroserviceCommunication{}
	err = json.Unmarshal([]byte(responseData), serverResponse)

	if err != nil {
		logger.Sugar().Error("Unmarshalling response did not go well: ", err)
	}

	accuracy := serverResponse.Data.GetFields()["accuracy"].GetNumberValue()
	gradientList := serverResponse.Data.GetFields()["gradients"].GetListValue().GetValues()

	gradients := []string{}
	for _, val := range gradientList {
		gradients = append(gradients, val.GetStringValue())
	}

	// TODO: Send the gradients back to the client to update their models
	index := 0
	for auth, url := range clients {
		wg.Add(1)
		target := strings.ToLower(auth)
		endpoint := fmt.Sprintf("http://%s:8080/agent/v1/vflTrainRequest/%s", url, target)

		dataRequest["type"] = "vflGradientDescentRequest"
		dataRequest["data"] = map[string]any{
			"gradients":     gradients[index],
			"learning_rate": learning_rate,
		}

		index++

		dataRequestJson, err := json.Marshal(dataRequest)
		if err != nil {
			logger.Sugar().Errorf("Error marshalling combined data: %v", err)
			return 0., err
		}

		go func() {
			response, err := sendData(endpoint, dataRequestJson)
			if err != nil {
				logger.Sugar().Error("Error sending data, ", err, ", received: ", response)
			}
			wg.Done()
		}()
	}

	wg.Wait()

	return accuracy, nil
}

func runVFLTraining(dataRequest map[string]any, authorizedProviders map[string]string, jobId string, ctx context.Context, requestID string) []byte {
	clients := map[string]string{}
	var serverUrl string
	var serverAuth string
	var finalAccuracy float64
	var wg sync.WaitGroup

	var cycles int64 = 10
	var learning_rate float64 = 0.05
	var policy_removal int64 = -1
	var policy_reintroduction int64 = -1
	var dataProviders []string = []string{}

	var trainingBacktrack int64 = 0 // Default value

	data, ok := dataRequest["data"].(map[string]any)
	logger.Sugar().Info("Data from req: ", data)

	if ok {
		floatCycles, ok := data["cycles"].(float64)

		if ok {
			cycles = int64(floatCycles)
		}

		floatLearningRate, ok := data["learning_rate"].(float64)
		if ok {
			learning_rate = floatLearningRate
		}

		trainingBacktrackVal, ok := data["training_backtrack"].(float64)
		if ok {
			trainingBacktrack = int64(trainingBacktrackVal)
		} else {
			logger.Sugar().Debug("training_backtrack not set, defaulting to: ", trainingBacktrack)
		}

		policyRemoval, ok := data["policy_removal"].(float64)
		if ok {
			policy_removal = int64(policyRemoval)
		}

		reintroducePolicies, ok := data["policy_reintroduction"].(float64)
		if ok {
			policy_reintroduction = int64(reintroducePolicies)
		}

		logger.Sugar().Debug("policy_removal round: ", policy_removal, ", policy_reintroduction round: ", policy_reintroduction)
	}

	metadata := map[string]any{
		"total_rounds":          cycles,
		"policy_removal":        policy_removal,
		"training_backtrack":    trainingBacktrack,
		"policy_reintroduction": policy_reintroduction,
	}

	// logger.Sugar().Debug("metadata: ", metadata)

	var results []map[string]any
	trainingFailed := false

	for auth, url := range authorizedProviders {
		if strings.ToLower(auth) == "server" {
			serverUrl = url
			serverAuth = auth
		} else if url != "" {
			clients[auth] = url
		}

		dataProviders = append(dataProviders, auth)
	}

	logger.Sugar().Info("Sending ping to start pods...")
	dataRequest["type"] = "vflPingRequest"

	dataRequestJson, err := json.Marshal(dataRequest)
	if err != nil {
		logger.Sugar().Errorf("Error marshalling combined data: %v", err)
		return []byte{}
	}

	user, ok := dataRequest["user"].(*pb.User)

	if !ok {
		logger.Sugar().Info("Did not retrieve User from dataRequest, cannot dynamically verify each training round.")
		user = &pb.User{}
	}

	var noPing bool = false

	for auth, url := range authorizedProviders {
		wg.Add(1)
		target := strings.ToLower(auth)
		endpoint := fmt.Sprintf("http://%s:8080/agent/v1/vflTrainRequest/%s", url, target)

		go func() {
			// TODO: Repeat ping until no error, after 5 tries, cancel request
			for i := 0; i < 5; i++ {
				_, err := sendData(endpoint, dataRequestJson)
				if err == nil {
					break
				}
				time.Sleep(300 * time.Millisecond)
				if i == 4 {
					noPing = true
				}
			}

			wg.Done()
		}()
	}

	// note: this is likely misplaced, probably needs to be after the wait
	if noPing {
		logger.Sugar().Error("No ping from a client or the server. Something is wrong.")
	}

	wg.Wait()

	logger.Sugar().Info("Running VFL for ", cycles, " rounds")
	for round := range cycles {
		logger.Sugar().Info("Running VFL training round ", round)

		numClients := -1          // default value in case of error
		metadata_accuracy := -1.0 // default value in case of error

		// TODO: Implement policy change request
		if policy_removal == round {
			logger.Sugar().Info("Sending in the policy change request, removing client 3 from the agreement.")
			logger.Sugar().Info("TODO: Policy change request not yet implemented.")

			policyUpdate := &pb.RequestApproval{
				Type:             "policyRemoval",
				User:             user,
				DestinationQueue: "policyEnforcer-in",
			}

			// Create a channel to receive the response
			responseChan := make(chan validation)

			requestApprovalMutex.Lock()
			requestApprovalMap[policyUpdate.User.Id] = responseChan
			requestApprovalMutex.Unlock()

			logger.Sugar().Info("- Sending policy removal request")
			_, err = c.SendRequestApproval(ctx, policyUpdate)
			if err != nil {
				logger.Sugar().Warnf("error in sending/receiving policy removal: %v", err)
			}
		}

		// TODO: Implement policy change request
		if policy_reintroduction == round {
			logger.Sugar().Info("Sending in the policy change request, reintroducing client 3 to the agreement.")
			logger.Sugar().Info("TODO: Policy change request not yet implemented. (values are hardcoded)")

			policyUpdate := &pb.RequestApproval{
				Type:             "policyReintroduction",
				User:             user,
				DestinationQueue: "policyEnforcer-in",
			}

			// Create a channel to receive the response
			responseChan := make(chan validation)

			requestApprovalMutex.Lock()
			requestApprovalMap[policyUpdate.User.Id] = responseChan
			requestApprovalMutex.Unlock()

			logger.Sugar().Info("- Sending policy reintroduction request")
			_, err = c.SendRequestApproval(ctx, policyUpdate)
			if err != nil {
				logger.Sugar().Warnf("error in sending/receiving policy reintroduction: %v", err)
			}
		}

		protoRequest := &pb.RequestApproval{
			Type:             "vflTrainModelRequest",
			User:             user,
			DataProviders:    dataProviders,
			DestinationQueue: "policyEnforcer-in",
		}

		// Create a channel to receive the response
		responseChan := make(chan validation)

		requestApprovalMutex.Lock()
		requestApprovalMap[protoRequest.User.Id] = responseChan
		requestApprovalMutex.Unlock()

		noValidation := false

		logger.Sugar().Info("- Sending policy reverification request")
		for i := range 5 {
			_, err = c.SendRequestApproval(ctx, protoRequest)
			if err != nil {
				logger.Sugar().Warnf("error in sending/receiving requestApproval: %v", err)
			}

			if err == nil {
				// on success we can continue 
				break
			}

			if i == 4 {
				noValidation = true
			}
		}

		if noValidation {
			logger.Sugar().Error("No reverification approval received, error in network. Shutting down operation.")
			trainingFailed = true
			break
		}

		select {
		case validationStruct := <-responseChan:
			msg := validationStruct.response
			logger.Sugar().Info("Received validation message: ", msg, ", with vstruct: ", validationStruct)

			if msg.Type != "requestApprovalResponse" {
				logger.Sugar().Errorf("Unexpected message received, type: %s", msg.Type)
				return []byte{}
			}

			if msg.Error != "" {
				logger.Sugar().Info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
				logger.Sugar().Info("   Policy does not allow this training to continue.")
				logger.Sugar().Info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
				break
			}

			// logger.Sugar().Debug("AuthorizedProviders from policy response: ", msg.AuthorizedProviders)
			// logger.Sugar().Debug("AuthorizedProviders originally requested: ", authorizedProviders)
			// logger.Sugar().Debug("Current clients before sync: ", clients)
			if len(msg.AuthorizedProviders) != len(authorizedProviders) {

				// if len is different I can still allow training to continue with the authorised ones
				// in that case remove the unauthorised ones from the clients map
				// or add the authorised ones if they were not present before
				for auth_provider := range authorizedProviders {
					if _, ok := msg.AuthorizedProviders[auth_provider]; !ok {
						logger.Sugar().Debug("Removing unauthorised provider: ", auth_provider, " from the training.")
						delete(clients, auth_provider)
					}
				}
			}

			// maybe we can merge the above if into this one
			if len(clients) != len(authorizedProviders) {
				// add newly authorised clients that are not yet in the clients map
				for auth_provider, url := range authorizedProviders {
					if strings.ToLower(auth_provider) != "server" { // exclude server
						if _, ok := msg.AuthorizedProviders[auth_provider]; ok {
							if _, exists := clients[auth_provider]; !exists {
								logger.Sugar().Debug("Adding newly authorised provider: ", auth_provider, " to the training.")
								clients[auth_provider] = url
							}
						}
					}
				}
			}

			logger.Sugar().Debug("Clients: ", clients)
			numClients = len(clients)

			logger.Sugar().Info("- Sending training request")
			accuracy, err := runVFLTrainingRound(dataRequest, clients, serverAuth, serverUrl, learning_rate, trainingBacktrack)
			logger.Sugar().Info("- Intermediate accuracy achieved: ", accuracy, " for round ", round)
			finalAccuracy = accuracy
			metadata_accuracy = accuracy // store accuracy from metadata for results

			if err != nil {
				logger.Sugar().Error("Training round returned an error.")
				trainingFailed = true
				break
			}
		}

		result := map[string]any{
			"timestamp":   time.Now().Format(time.RFC3339),
			"train_round": round,
			"accuracy":    metadata_accuracy,
			"clients":     numClients,
		}
		results = append(results, result)
		v, _ := trainingRequests.Load(requestID)
		reqData := v.(TrainingRequestData)
		reqData.Results = results
		trainingRequests.Store(requestID, reqData)
		
		if trainingFailed {
			break
		}

	}

	logger.Sugar().Info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
	logger.Sugar().Info("Final accuracy achieved: ", finalAccuracy)
	logger.Sugar().Info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

	dataRequest["type"] = "vflShutdownRequest"

	dataRequestJson, err = json.Marshal(dataRequest)
	if err != nil {
		logger.Sugar().Errorf("Error marshalling combined data: %v", err)
		return []byte{}
	}

	for auth, url := range authorizedProviders {
		wg.Add(1)
		target := strings.ToLower(auth)
		endpoint := fmt.Sprintf("http://%s:8080/agent/v1/vflTrainRequest/%s", url, target)

		go func() {
			sendData(endpoint, dataRequestJson)
			wg.Done()
		}()
	}

	wg.Wait()

	response := map[string]any{
		"jobId":    jobId,
		"accuracy": finalAccuracy,
	}

	logger.Sugar().Info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
	logger.Sugar().Info("Final accuracy achieved: ", finalAccuracy)
	logger.Sugar().Info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

	new_response := map[string]any{
		"metadata": metadata,
		"results":  results,
	}

	// --- SET STATUS and UNLOCK ---
	v, ok := trainingRequests.Load(requestID)
	if ok {
		reqData := v.(TrainingRequestData)
		reqData.Metadata = metadata
		if trainingFailed {
			reqData.Status = StatusFailed
		} else {
			reqData.Status = StatusDone
		}
		trainingRequests.Store(requestID, reqData)
		logger.Sugar().Infow("Job completed", "requestID", requestID, "status", reqData.Status, "accuracy", finalAccuracy)
	} else {
		logger.Sugar().Error("Could not find the training request to update status.")
	}

	// Release the active job lock
	activeJobLock.Lock()
	activeJobID = ""
	activeJobLock.Unlock()

	// Marshal and return
	responseJson, err := json.MarshalIndent(new_response, "", "    ")
	if err != nil {
		logger.Sugar().Errorf("Error marshalling training results: %v", err)
		return []byte{}
	}

	logger.Sugar().Info("Training results: ", string(responseJson))
	return cleanupAndMarshalResponse(response)  // note this is not the same as responseJson
}

// Use the data request that was previously built and send it to the authorised providers
// acquired from the request approval
func sendDataToAuthProviders(dataRequest []byte, authorizedProviders map[string]string, msgType string, jobId string) []byte {
	// Setup the wait group for async data requests
	var wg sync.WaitGroup
	var responses []string

	// This will be replaced with AMQ in the future
	agentPort := "8080"
	// Iterate over each auth provider
	for auth, url := range authorizedProviders {
		wg.Add(1)
		target := strings.ToLower(auth)
		// Construct the end point
		endpoint := fmt.Sprintf("http://%s:%s/agent/v1/%s/%s", url, agentPort, msgType, target)

		logger.Sugar().Infof("Sending request to %s.\nEndpoint: %s\nJSON:%v", target, endpoint, string(dataRequest))

		// Async call send the data
		go func() {
			respData, err := sendData(endpoint, dataRequest)
			if err != nil {
				logger.Sugar().Errorf("Error sending data, %v", err)
			}
			responses = append(responses, respData)
			// Signal that the data request has been sent to all auth providers
			wg.Done()
		}()
	}

	// Wait until all the requests are complete
	wg.Wait()
	logger.Sugar().Debug("Returning responses")

	responseMap := map[string]any{
		"jobId":     jobId,
		"responses": responses,
	}

	// jsonResponse, _ := json.Marshal(responseMap)
	// return jsonResponse
	return cleanupAndMarshalResponse(responseMap)
}

// Now assumes input is map[string]interface{} and directly marshals it to prettified JSON.
func cleanupAndMarshalResponse(responseMap map[string]any) []byte {
	prettifiedJSON, err := json.MarshalIndent(responseMap, "", "    ")
	if err != nil {
		logger.Sugar().Errorf("Error marshalling cleaned response: %v", err)
	}
	return prettifiedJSON
}

func sendData(endpoint string, jsonData []byte) (string, error) {
	// FIXME: Change to an actual token in the future?
	headers := map[string]string{
		"Authorization": "bearer 1234",
	}
	body, err := api.PostRequest(endpoint, string(jsonData), headers)
	if err != nil {
		return "", err
	}

	return string(body), nil
}

func availableProvidersHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		logger.Debug("Starting requestApprovalHandler")
		var availableProviders = make(map[string]lib.AgentDetails)
		resp, err := getAvailableProviders()
		if err != nil {
			logger.Sugar().Errorf("Error getting available providers: %v", err)
			return
		}

		// Bind resp to availableProviders
		availableProviders = resp

		jsonResponse, err := json.Marshal(availableProviders)
		if err != nil {
			logger.Sugar().Errorf("Error marshalling result, %v", err)
			http.Error(w, "Internal server error", http.StatusInternalServerError)
			return
		}

		w.WriteHeader(http.StatusOK)
		w.Write(jsonResponse)
	}
}

// Maybe this should be moved into the orchestrarot
func getAvailableProviders() (map[string]lib.AgentDetails, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Get the value from etcd.
	resp, err := etcdClient.Get(ctx, "/agents/online", clientv3.WithPrefix())
	if err != nil {
		logger.Sugar().Errorf("failed to get value from etcd: %v", err)
		return nil, err
	}

	// Initialize an empty map to store the unmarshaled structs.
	result := make(map[string]lib.AgentDetails)
	// Iterate through the key-value pairs and unmarshal the values into structs.
	for _, kv := range resp.Kvs {
		var target lib.AgentDetails
		err = json.Unmarshal(kv.Value, &target)
		if err != nil {
			// return nil, fmt.Errorf("failed to unmarshal JSON for key %s: %v", key, err)
		}
		result[string(target.Name)] = target
	}

	return result, nil

}
