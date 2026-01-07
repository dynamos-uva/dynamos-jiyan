#!/bin/bash

{
if [ -z "$1" ]; then
  echo "No namespace name provided."
  exit 1
fi

if ! grep -q "name: $1" charts/namespaces/templates/namespaces.yaml; then
tee -a charts/namespaces/templates/namespaces.yaml << END
---

apiVersion: v1
kind: Namespace
metadata:
  name: $1
  annotations:
    "helm.sh/resource-policy": keep
    "app.kubernetes.io/managed-by": "Helm"
    "config.linkerd.io/trace-collector": collector.linkerd-jaeger:55678 # or 14268?

---

apiVersion: v1
kind: Secret
metadata:
  name: rabbit
  namespace: $1
type: Opaque
data:
  password: {{ .Values.secret.password | b64enc | quote }}
END
fi
}
