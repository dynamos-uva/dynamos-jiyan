# PET enabled vertical federated learning in DYNAMOS

This fork of the original DYNAMOS project serves as a codebase for my bachelor thesis.
This thesis extends the original DYNAMOS VFL workflow by adding in two new privacy enhancing microservices, Anonymization and Differential Privacy.

The new microservices and datasets can be found in the following directories:

- [Anonymization Microservice](python/anonymize-test)
- [Differential Privacy Microservice](python/differential-p-test)

To run the VFL training with the new microservices enabled, follow the same installation steps for the [regular DYNAMOS setup](https://github.com/Jorrit05/DYNAMOS/tree/main).
To test the deployment, an example VFL training request can be made.

Using curl, it could look like this:
```
curl -H "Host: api-gateway.api-gateway.svc.cluster.local" \
     -H "Content-Type: application/json" \
     --data-raw '{
       "type": "vflTrainModelRequest",
       "user": {
         "id": "1234",
         "userName": "jake.jongejans@student.uva.nl"
       },
       "dataProviders": [
         "server",
         "clientone",
         "clienttwo",
         "clientthree"
       ],
       "data_request": {
         "type": "vflTrainModelRequest",
         "data": {
           "learning_rate": 0.1,
           "cycles": 180,
           "policy_removal": -1,
           "policy_reintroduction": -1,
           "training_backtrack": 0
         },
         "requestMetadata": {}
       }
     }' \
     "http://api-gateway.api-gateway.svc.cluster.local:80/api/v1/requestApproval"
```
