# Recommender system trained on movielens-100k dataset
the recommender system employs two type of models. basic large model and small model. Both are able to make predictions using ONNX. Even though
the model can be transpiled and completely supported in Orion, but the model cannot be compiled and deployed because of Cairo programming language memory allocation
problems. the issue has been address in the [discord thread](https://discord.com/channels/1070370565761802372/1210497513208160266).

the model is stepping stone to develop further recommender systems for social media frameworks. Like web3 movie streaming websites.
