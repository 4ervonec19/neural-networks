#### Постановка ударений в слове с использованием архитектур трансформера

В текущем репозитории решалась задача постановки ударения в слове, подававшемся на вход модели, например:

#### человек -> model(человек) -> человЕк

Были обучены два трансформера из семейства BERT для классификации ударной позиции в слове и получены соответствующие файлы с реализциями подходов:

* BERT ($accuracy \approx 0.95$)
    * [BERT_train](transformer_learn_BERT_notebook.ipynb)
    * [BERT_inference](transformer_learn_BERT_with_inference.ipynb)
 
* DaBERTa ($accuracy \approx 0.968$)
    * [DaBERTa_train](transformer_learn_DaBERTa.ipynb)
    * [DaBERTa_inference](transformer_inference_DaBERTa.ipynb)

Так же имеется файл с [логами](transformer_training_logs.png) обучения BERT. 

Для ознакомления с логами обучения DaBERTa перейдите по ссылке [¡тык!](https://wandb.ai/4ervonec19-bauman-moscow-state-technical-university/DaBERTa_Accents_GO/workspace?nw=nwuser4ervonec19).


#### Ermine is watching you...

![ermine](ermine_is_watching_you.jpg)
