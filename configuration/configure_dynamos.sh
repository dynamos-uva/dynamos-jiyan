#!/bin/bash

{
# Parse the agents and thirdparties from the CLI arguments
IFS=',' read -r -a agents <<< "$1"
IFS=',' read -r -a thirdparties <<< "$2"

cd DYNAMOS

echo "Adding agents..."
for agent in "${agents[@]}"
do
    echo "- agent '$agent'"
    ./scripts/add_agent.sh $agent > /dev/null
done

echo ""
echo "Adding third parties..."
for thirdparty in "${thirdparties[@]}"
do
    echo "- third party '$thirdparty'"
    ./scripts/add_thirdparty.sh $thirdparty > /dev/null
done
}