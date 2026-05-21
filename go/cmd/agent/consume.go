package main

import (
	"context"
	"fmt"

	"github.com/Jorrit05/DYNAMOS/pkg/etcd"
	"github.com/Jorrit05/DYNAMOS/pkg/lib"
	pb "github.com/Jorrit05/DYNAMOS/pkg/proto"
	"google.golang.org/protobuf/types/known/anypb"
)

func handleIncomingMessages(ctx context.Context, grpcMsg *pb.SideCarMessage) error {
	logger.Sugar().Info("Running handleIncomingMessages.")
	ctx, span, err := lib.StartRemoteParentSpan(ctx, serviceName+"/func: handleIncomingMessages/"+grpcMsg.Type, grpcMsg.Traces)
	if err != nil {
		logger.Sugar().Warnf("Error starting span: %v", err)
	}
	defer span.End()

	switch grpcMsg.Type {
	case "compositionRequest":
		logger.Debug("Received compositionRequest")

		compositionRequest := &pb.CompositionRequest{}

		if err := grpcMsg.Body.UnmarshalTo(compositionRequest); err != nil {
			logger.Sugar().Errorf("Failed to unmarshal compositionRequest message: %v", err)
		}
		go compositionRequestHandler(ctx, compositionRequest)
	case "microserviceCommunication":
		handleMicroserviceCommunication(ctx, grpcMsg)
	case "Request":
		// handleRequestDataProvider
		// Receive Request through RabbitMQ, means we received the request from the computeProvider
		// Implicitly this means I am only a dataProvider
		logger.Debug("Received Request from Rabbit (third party)")

		request := &pb.Request{}

		if err := grpcMsg.Body.UnmarshalTo(request); err != nil {
			logger.Sugar().Errorf("Failed to unmarshal Result message: %v", err)
		}

		ttpMutex.Lock()
		thirdPartyMap[request.RequestMetadata.CorrelationId] = request.RequestMetadata.ReturnAddress
		ttpMutex.Unlock()

		msComm := &pb.MicroserviceCommunication{}
		msComm.RequestMetadata = &pb.RequestMetadata{}

		msComm.Type = "microserviceCommunication"
		msComm.RequestType = request.Type
		// Set own routing key as return address to ensure the response comes back to me and then returned to where it needs
		msComm.RequestMetadata.ReturnAddress = agentConfig.RoutingKey
		msComm.RequestMetadata.CorrelationId = request.RequestMetadata.CorrelationId

		any, err := anypb.New(request)
		if err != nil {
			logger.Sugar().Error(err)
			return err
		}

		msComm.OriginalRequest = any
		compositionRequest, err := getCompositionRequest(request.User.UserName, request.RequestMetadata.JobId)
		if err != nil {

			logger.Sugar().Errorf("Error getting matching composition request: %v", err)
			return err
		}
		msComm.RequestMetadata.DestinationQueue = compositionRequest.LocalJobName
		key := fmt.Sprintf("/agents/jobs/%s/queueInfo/%s", serviceName, compositionRequest.LocalJobName)
		value := compositionRequest.LocalJobName

		// No options
		generateChainAndDeploy(ctx, compositionRequest, compositionRequest.LocalJobName, request.Options)
		c.SendMicroserviceComm(ctx, msComm)

		logger.Sugar().Warnf("key: %v", key)
		logger.Sugar().Warnf("value: %v", value)
		err = etcd.PutEtcdWithGrant(ctx, etcdClient, key, value, queueDeleteAfter)
		if err != nil {
			logger.Sugar().Errorf("Error PutEtcdWithGrant: %v", err)
		}
	default:
		logger.Sugar().Errorf("Unknown message type: %s", grpcMsg.Type)
		return fmt.Errorf("unknown message type: %s", grpcMsg.Type)
	}

	return nil
}
