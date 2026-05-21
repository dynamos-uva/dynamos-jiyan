#!/bin/bash

API_GATEWAY_POD=$(kubectl get pods -n api-gateway | grep api-gateway | sed "s/^\(api-gateway[a-zA-Z0-9-]\+\).*/\1/")
logs=$(kubectl logs $API_GATEWAY_POD -c api-gateway -n api-gateway | sed "s/\t/ /")

intermediate_accuracies=$(echo "$logs" | awk '
  /Intermediate accuracy achieved:/ {
    for (i = 1; i <= NF; ++i) {
      if ($i ~ /^-?[0-9]+(\.[0-9]+)?$/) {
        print $i
        break
      }
    }
  }')

final_accuracy=$(echo "$logs" | awk '
  /Final accuracy achieved:/ {
    for (i = NF; i >= 1; --i) {
      if ($i ~ /^-?[0-9]+(\.[0-9]+)?$/) {
        print $i
        break
      }
    }
  }')

filename="${1:-default}_intermediate_results.txt"

mkdir -p results
# echo "${intermediate_accuracies[@]}" | tr " " "\n" > "results/$filename"
echo "$intermediate_accuracies" | tr " " "\n" 
echo "$intermediate_accuracies" | tr " " "\n" > "results/$filename"
