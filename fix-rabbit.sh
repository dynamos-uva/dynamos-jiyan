#!/bin/bash

# Get pw kubernetes uses
PW=$(kubectl -n api-gateway get secret rabbit -o jsonpath='{.data.password}' | base64 -D)

# Get rabbitmq pod
RABBIT_POD=$(kubectl -n core get pods -l app=rabbitmq -o jsonpath='{.items[0].metadata.name}')

# Change pw to match
kubectl -n core exec "$RABBIT_POD" -c rabbitmq -- rabbitmqctl change_password normal_user "$PW"

# Delete all pods to restart
kubectl -n orchestrator delete pod -l app=orchestrator
kubectl -n orchestrator delete pod -l app=policy-enforcer
kubectl -n vu delete pod -l app=vu
kubectl -n uva delete pod -l app=uva
kubectl -n surf delete pod -l app=surf
kubectl -n api-gateway delete pod -l app=api-gateway
kubectl -n server delete pod -l app=server
kubectl -n clientone delete pod -l app=clientone
kubectl -n clienttwo delete pod -l app=clienttwo
kubectl -n clientthree delete pod -l app=clientthree