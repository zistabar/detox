# detox

Identification and classification of toxic comments using machine learning techniques.

Links to the pre-trained models used in the application:

- GloVe: <http://nlp.stanford.edu/data/glove.6B.zip>;
- fastText: <https://fasttext.cc/docs/en/english-vectors.html> (select **_wiki-news-300d-1M.vec.zip_** file).

Place the pre-trained models in the following directory: `/pretrained-models/*`.

A container with the created model and REST API that returns predictions for a given comment is in the following repository: <https://hub.docker.com/r/zista/detox-comments-v1.1>.
To run the container locally, run the following command:

`sudo docker run -p 5000:5000 zista/detox-comments-v1.1:azure-deploy`

An example request that the created REST API can take to predict a comment looks as follows:

```shell
curl --location --request POST 'http://detox-comments.azurewebsites.net/detox/api/comment' \
--header 'Content-Type: application/json' \
--data-raw '{
    "comment": "Foo bar"
}'
```
