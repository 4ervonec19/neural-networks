#### ASR Inference

В данной работе рассматривается инференс датасета, содержащего .wav файлы для получения метрики $WER$, определяющей качество транскрибации ASR-модели.

#### Файлы

* [asr_inference_first_look.ipynb](asr_inference_first_look.ipynb) - файл с первоначальным взглядом на данные и неакцентированной попыткой применить whisper_large_v3 модель.

* [asr_inference_large_whisper_submit.ipynb](asr_inference_large_whisper_submit.ipynb) - файл с пристальным взглядом на whisper_large_v3 с улучшением в виде Voice Actitvity Detection (VAD) модели Silero-VAD. Результат: $WER = 0.257$.

* [GigaAM_CTC_inference.ipynb](GigaAM_CTC_inference.ipynb) - файл с попыткой инференса модели от Sber на основе CTCLoss, затюненной под русский язык. $WER = 0.228$.

* [GigaAM_RNNT_inference.ipynb](GigaAM_RNNT_inference.ipynb) - файл с попыткой инференса модели от Sber на основе RNNT, затюненной под русский язык. $WER = 0.0783$. Дополнительно файл преобразуется в формат .parquet для учёта пустой строки. В случае .csv формата получаем NaN при открытии сохраненного файла. Подход с .parquet позволил сделать скачок в качестве предсказания на соревновании.

* [data.parquet](data.parquet) - финальное предсказание (submit).

* [kaggle.json](kaggle.json) - файл для запуска в GoogleCollab (перебрасывает данные из kaggle в GoogleCollab)