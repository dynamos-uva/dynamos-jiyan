#!/bin/bash

{
if [ -z "$1" ]; then
  echo "No agent name provided."
  exit 1
fi

sed -e "s/^# //" -e "s/%WORKER%/$1/g" "charts/agents/templates/workerX.yaml" > "charts/agents/templates/${1}.yaml"

if ! grep -q "namespace: $1" charts/agents/templates/cluster_role.yaml; then
tee -a charts/agents/templates/cluster_role.yaml << END
---

apiVersion: v1
kind: ServiceAccount
metadata:
  name: job-creator-$1
  namespace: $1
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: job-creator-$1
  namespace: $1
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: job-creator
subjects:
- kind: ServiceAccount
  name: job-creator-$1
  namespace: $1
END
fi

./scripts/add_namespace.sh $1
}
