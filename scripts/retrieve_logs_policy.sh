#!/bin/bash

POLICY_POD=$(kubectl get pods -n orchestrator | grep policy-enforcer | sed "s/^\(policy-enforcer[a-zA-Z0-9-]\+\).*/\1/")
logs=$(kubectl logs $POLICY_POD -c policy-enforcer -n orchestrator | sed "s/\t/ /")

echo "$logs"