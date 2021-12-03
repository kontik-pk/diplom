## VGG

`VGG-19` — сверточная нейронная сеть (CNN), которая имеет 19 главных слоев (16 сверточных, 3 полносвязных) а также 5 слоев `MaxPool` и 1 слой `SoftMax`. Она была сконструирована и обучена в Оксфордском университете в 2014 году. Для обучения сети VGG-19 использовалось более чем 1 миллиона изображений из базы данных `ImageNet`. Эта предварительно обученная сеть может классифицировать до 1000 объектов.

Сеть VGG впервые представлена в [этой статье](https://arxiv.org/pdf/1409.1556v6.pdf). Авторы экспериментировали с глубиной сверточной сети и изучали влияние количества слоев в сети на итоговое качество классификации. На вход сетям подавались изображения 224 × 224 пикселя RGB. Единственная предварительная обработка - это вычитание из каждого пикселя среднего значения RGB, вычисленного на обучающем наборе. Каждое изображение проходило через сверточные слои, размер свертки составлял 3×3 пикселя, шаг свертки - 1 пиксель. 
За сверточными слоями следует три полносвязных слоя. Первые два содержат по 4096 нейронов, а последний - 1000 нейронов (по количеству определяемых классов). Архитектура сети завершается слоем `SoftMax`. В качестве функции активации используется `ReLU`.

Обучение сети осуществлялось с помощью градиентного спуска, размер батча составлял 256 изображений. Скорость обучения изначально установлена 10^2, но уменьшалась в 10 раз, как только качество классифкации на валидационной выборке переставало улучшаться. Скорость обучения уменьшалась 3 раза, а обучение остановлено на 74 эпохах. 

Авторами исследовались сети глубиной 11, 13, 16 и 19 слоев. Лучшие результаты показала самая глубокая сеть.

В настоящий момент, сеть `VGG-19` занимает 436 место по качеству классификации на датасете `ImageNet` (top 1 accuracy - 74.5%, top 5 accuracy -	92.0%). Рейтинг доступен [здесь](https://paperswithcode.com/sota/image-classification-on-imagenet).

## ResNet

ResNet — сокращенное название для Residual Network (дословно  — «остаточная сеть»). Сеть ResNet была разработана в Microsoft в 2015 году для решения задачи распознавания изображений. Эта модель также обучена на более чем 1 миллионе изображений из базы данных `ImageNet`. ResNet может классифицировать до 1000 объектов, принимает на вход цветные изображения размером 224×224 пикселей. Данная сеть была разработана с целью избавиться от затухающих и взрывных градиентов.

[Статья](https://arxiv.org/pdf/1512.03385v1.pdf), в которой представлена сеть `ResNet` начинается с вопроса о том, всегда ли увеличение глубины сети приводит к лучшему результату. Ответ на этот вопрос, конечно, отрицательный, поскольку увеличение количества слоев приводит к проблеме затухающих/взрывающихся градиентов. Из-за этого обучение глубоких сетей затрудняется - поскольку градиент распространяется обратно на более ранние слои, повторное умножение может сделать градиент бесконечно малым. В результате, по мере того, как сеть углубляется, качество классификации начинает быстро ухудшаться. Для решения данной проблемы, авторы предлагают использовать *остаточное обучение*, основная особенность которого - использование *остаточных блоков* (*Residual blocks*) в архитектуре модели.

Нейронная сеть ResNet-152, представленная в 2015 году, в настоящий момент занимает 339 место в соревновании по класстификации изображений на датасете `ImageNet`. Однако с 2015 года предпринято множество успешных попыток модифицировать архитектуру, и ее различные реализации находятся в рейтинге моделей гораздо выше оригинала.

### Residual blocks

При увеличении количества слоев в нейронной сети, качество обучения растет до определенного момента, а потом начинает уменьшаться. Причиной этому может быть как переобучение (нейросеть "запоминает" признаки, что приводит к великолепным результатам на обучающей выборке, и к ухудшению качества на валидационной), так и затухание градиента. Авторы [статьи](https://arxiv.org/pdf/1512.03385v1.pdf) также показали, что более глубокие сети могут обучаться хуже, чем неглубокие. Такую проблему деградации можно решить, модифицировав процесс обучения. Архитектуры сетей до появления `ResNet` представляли собой последовательность слоев, через которые пропускается исходное изображение. После того, как исходное изображение пропустится через один слой, результат преобразования отправляется дальше, к следующему слою, и так далее. 
Идея `residual blocks` основана на том, чтобы сохранять "остаточную" информацию перед переходом к следующим слоям. Например, назовем входную матрицу `x`, а наша цель - найти оптимальное распределение весов в свертке `H(x)`. Тогда разница между входом и выходом (или остаток) будет:
```
R(x) = Output — Input = H(x) — x
```
А искомое распределение

```
H(x) = R(x) + x
```

На настоящий момент `DenseNet-121` занимает 426 место в соревновании по классификации изображений на датасете `ImageNet`, более глубокий аналог `DenseNet-169` занял 399 место, `DenseNet-201` поднялся на 370 место, `DenseNet-264` - 362.


## DenseNet

DenseNet (Densely Connected Convolutional Network) была предложена в 2017 году в статье [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993v5.pdf). Успех ResNet (Deep Residual Network) позволил предположить, что укороченное соединение в CNN позволяет обучать более глубокие и точные модели. Авторы проанализировали это наблюдение и представили компактно соединенный (dense) блок - все слои (с соответствующими размерами карты признаков) соединены напрямую друг с другом, то есть каждый слой получает дополнительные входные данные от всех предыдущих слоев и передает свои собственные карты признаков всем последующим слоям. Важно отметить, что, в отличие от ResNet, признаки, прежде чем они будут переданы в следующий слой, не суммируются, а конкатенируются в единый тензор. При этом количество параметров сети DenseNet намного меньше, чем у сетей с такой же точностью работы. Авторы утверждают, что DenseNet работает особенно хорошо на малых наборах данных.

Если общее количество соединений в архитектуре более ранних сетей равно количеству слоев `L`, то, поскольку в `DenseNet` каждый слой связан со всеми предыдущими, количество соединений представленной архитектуре равно `L * (L + 1) / 2`.

Благодаря использованию конкатенации карт признаков, количество параметров архитектуры `DenseNet` меньше, чем в более ранних архитектурах с аналогичным количеством слоев, в которых каждый слой читает состояние из предыдущего слоя и записывает на следующий. Фактически, количество параметров ResNets велико, потому что у каждого слоя есть свои веса, которые нужно обучать. Вместо этого слои `DenseNet` очень узкие (например, 12 фильтров), и они просто добавляют небольшой набор новых карт признаков. Иными словами, `DenseNet` хранит большую часть карт признаков с предыдущих слоев неизмененной, а часть дополняет "новыми знаниями", поэтому классификатор имеет возможность принимать решение, опираясь на информацию, полученную на каждом из слоев. Таким образом, основная мощь этой архитектуры состоит в повторном использовании карт признаков, которые не изменяются от слоя к слою, а дополняются новой информацией. Выразить выход Dense-блока можно следующим образом:

H(x<sub>n</sub>) = R([х<sub>0</sub>, x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n-1</sub>)




## EfficientNet

Модель EfficientNet была опубликована в 2019 году компанией Google и является одной из самых современных разработок (state-of-the-art ) [4]. Авторы модели выяснили взаимосвязь между точностью и размером модели. Так, они установили, что точность модели CNN увеличивается вместе с увеличением ширины (количества фильтров в каждом слое), глубины (количество слоев в модели) и разрешения (размер входного изображения). Тем не менее, увеличивая глубину, ширину и высоту слоев пропорционально степени N, вычислительные затраты увеличатся со степени 2 до степени N. Поэтому и были созданы разное семейство архитектур EfficientNet, которые имеют разное количество параметров. Существует 8 реализаций EfficientNet, отсчитывающихся от B0 до B7 по мере увеличения сложности архитектуры. Тем не менее, даже самый простой EfficientNetB0 показывает хорошие результаты. При наличие всего лишь 5,3 миллионов параметров, он обеспечивает точность 77,1% (Top-1), поэтому дообучение (fine-tuning) модели машинного обучения не займет много времени.

# Источники

- [VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://arxiv.org/pdf/1409.1556v6.pdf)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385v1.pdf)
- [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993v5.pdf)
- https://towardsdatascience.com/understanding-and-visualizing-densenets-7f688092391a
- https://medium.com/@bigdataschool
- https://neurohive.io/ru/vidy-nejrosetej/resnet-34-50-101/
- https://habr.com/ru/post/498168
- https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec
- https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035