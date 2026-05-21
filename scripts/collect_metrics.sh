#!/bin/bash

DYNAMOS_PORT=$(kubectl get svc -n ingress | grep "nginx-nginx-ingress-controller" | sed "s/.*80:\([0-9]*\)\/TCP.*/\1/")
DYNAMOS_IP=$(kubectl get nodes -o wide | grep dynamos | sed "s/.*\s\([0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\).*/\1/")

for number in "10-4" "1000-2" "1000-4" "250-4" "500-3"; do
    echo "Running rounds with $number cycles"
    for ((i=1; i<6; i++)); do
        if [ "$number" = "10-4" ]; then
          number = 10
          i = 4
        fi
        if [ "$number" = "1000-2" ]; then
          number = 1000
          i = 2
        fi
        if [ "$number" = "1000-4" ]; then
          number = 1000
          i = 4
        fi
        if [ "$number" = "250-4" ]; then
          number = 250
          i = 4
        fi
        if [ "$number" = "500-3" ]; then
          number = 500
          i = 3
        fi
        ./configuration/dynamos-configuration.sh > /dev/null

        # Sleep so the API gateway actually gets a message.
        sleep 15

        echo "- Running for $number rounds for the ${i}th time."

        {
            curl -H "Host: api-gateway.api-gateway.svc.cluster.local" http://$DYNAMOS_IP:$DYNAMOS_PORT/api/v1/requestApproval \
            --header 'Content-Type: application/json' \
            --data-raw "{
                \"type\": \"vflTrainModelRequest\",
                \"user\": {
                    \"id\": \"1234\",
                    \"userName\": \"jake.jongejans@student.uva.nl\"
                },
                \"dataProviders\": [\"server\", \"clientone\", \"clienttwo\", \"clientthree\"],
                \"data_request\": {
                    \"type\": \"vflTrainModelRequest\",
                    \"data\": {
                      \"learning_rate\": 0.05,
                      \"cycles\": 20
                      \"change_policies\": 10
                    },
            \"requestMetadata\": {}
                }
            }" > /dev/null
        }

        echo "- Curl timeout, waiting $((number * 12)) seconds..."
        sleep $((number * 12))

        echo "- Exporting the results..."
        ./scripts/retrieve_data.sh "$number-$i"

        sleep 5

        echo "- Uninstalling DYNAMOS for clean slate next round..."
        helm uninstall agents api-gateway core orchestrator namespaces prometheus thirdparties > /dev/null

        echo "- Waiting on final terminations for 30 seconds..."
        sleep 30
    done
done
