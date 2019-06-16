# detox

Identification and classification of toxic comments using machine learning techniques.

Download link to the pretrained models used in the application:
- GloVe: http://nlp.stanford.edu/data/glove.6B.zip;
- fastText: https://fasttext.cc/docs/en/english-vectors.html (choose **_wiki-news-300d-1M.vec.zip_** file).

All pretrained models should be placed in the following directory: /pretrained-models/*.

Container with the created model and REST API that returns predicitons for a given comment can be found in the following repository: https://hub.docker.com/r/zista/detox-comments-v1.1.
To run container locally execute following command:

`sudo docker run -p 5000:5000 zista/detox-comments-v1.1:azure-deploy`

The container was also deployed as Web App on Microsoft Azure server. It can be reached under the following URL: http://detox-comments.azurewebsites.net/.
Example request that the created REST API can take to predict a comment looks as follow:

`curl -i -H "Content-Type: application/json" -X POST -d "{\"comment\":\"Foo bar\"}" http://detox-comments.azurewebsites.net/detox/api/comment`
