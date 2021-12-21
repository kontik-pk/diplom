## ResT

описание архитектуры

## CycleMLP

описание архитектуры

## Conformer

В сверточной нейронной сети (_CNN_) операции свертки хороши для извлечения локальных признаков, но плохо справляются с составлением глобального представления изображения, с генерализацией. В трансформерах каскадные _self-attention_ модули могут фиксировать зависимости признаков на большом расстоянии, но, к сожалению, ухудшают детализацию локальных элементов.
Авторы статьи [Conformer: Local Features Coupling Global Representations for Visual Recognition](https://arxiv.org/pdf/2105.03889v1.pdf) представляют новую гибридную архитектуру нейронной сети, которая объединяет достоинства сверточных сетей и трансформеров. 

В настоящее время, различные реализации трансформеров активно используются для решения задач компьютерного зрения. 
Метод [_ViT_](https://arxiv.org/pdf/2010.11929v2.pdf) (_visual transformer_) создает последовательность токенов путем разделения каждого изображения на фрагменты с учетом местоположения и применяет каскадные блоки трансформера для извлечения параметризованных векторов в качестве визуальных представлений. Благодаря механизму _self-attention_ и структуре многослойного персептрона (_MLP_), _visual transformer_ отражает сложные пространственные преобразования и зависимости признаков на большом расстоянии, которые составляют глобальные представления. К сожалению, _visual transformer_ не восприимчив к локальным особенностям признаков, что приводит к снижению различимости фона и объекта. Улучшенные вариации _visual transformer_ используют модуль токенизации или карты функций _CNN_ в качестве входных токенов для получения и использования информации о соседних элементах. Тем не менее, вопрос остается открытым, каким образом наиболее оптимально распознавать локальные признаки, не теряя глобального представления об объекте.

Архитектура гибридной нейронной сети `Conformer` содержит две ветви: первая ветвь представляет собой сверточную нейронную сеть семейства `ResNet`, а вторая - нейронная сеть семейства `ViT`. Ветви соединены между собой элементами _Feature Coupling Unit (FCU)_. Основная цель этих элементов - обеспечить обмен семантической информацией между блоками сетей этих ветвей.  Элементы _FCU_ разработаны с учетом размерности карт признаков _CNN_ и выхода трансформера: _FCU_ используют свертку _1×1_ для определения количества каналов, стратегии _down/up sampling_ для определения разрешения карты признаков, _LayerNorm_ и _BatchNorm_ для нормализации значений признаков. Поскольку _CNN_ и трансформер имеют тенденцию распознавать признаки на разных уровнях (локальный или глобальный), _FCU_ вставляется в каждый блок, чтобы последовательно устранить семантическое расхождение между ними в интерактивном режиме. Такая процедура слияния может значительно повысить способность глобального восприятия локальных признаков и распознавание локальных деталей в глобальных представлениях.


<div align="center">
  <img src="https://github.com/kontik-pk/diplom/blob/main/nets/transformers/illustrations/Comparison_of_feature_maps.png" width="1100" />
</div></br>
<div align="center">
  <figcaption>Рис. 1. <i>Сравнение карт признаков CNN (ResNet-101), Visual Transformer (DeiT-S) и Conformer. В то время как _CNN_ активирует отличительные локальные области (например, голова павлина на (a) и хвост на (e)), ветвь _CNN_ в Conformer использует информацию о глобальном представлении от Visual Transformer  и, таким образом, активирует весь объект (например, форма павлина на (b) и (f)). По сравнению с CNN, локальные особенности Visual Transformer ухудшаются (например, (c) и (g)). Напротив, ветвь трансформера в Conformer сохраняет детали локальных признаков из CNN, различая при этом фон (например, контуры павлина на (d) и (h) более четкие, чем на (c) и (g)) </i> </figcaption>
</div></br>

На Рисунке 1 отображена способность нейросетевой модели `Conformer` различать локальные признаки и составлять глобальное представление по сравнению с обычными _CNN_ и трансформерами. В то время как обычные _CNN_ (например, `ResNet-101`) имеют тенденцию улавливать отличительные локальные области (например, голову или хвост павлина),  ветвь _CNN_ модели `Conformer` может активировать весь экстент объекта (рис. 1 (b) и (f)). При использовании только трансформеров для слабо-различимых локальных особенностей (например, размытых границ объекта) трудно отличить объект от фона (рис. 1 (c) и (g)). Связь локальных признаков и глобального представления значительно повышает различимость признаков, полученных только с помощью трансформера (рис. 1 (d) и (h)).

Архитектура `Conformer` выглядит следующим образом (Рисунок 2):

<div align="center">
  <img src="https://github.com/kontik-pk/diplom/blob/main/nets/transformers/illustrations/Network_architecture_of_the_proposed_Conformer.png" width="1000" />
</div></br>
<div align="center">
  <figcaption>Рис. 2. <i> Архитектура сети Conformer. (a) Повышающая и понижающая дискретизация для пространственного выравнивания карт функций. (b) Детали реализации блока CNN, блока трансформера и блока Feature Coupling Unit (FCU). (c) Миниатюра Conformer</i> </figcaption>
</div></br>

В `Conformer` из ветви трансформера глобальное представление последовательно передается в карты признаков, чтобы усилить способность глобального восприятия ветви _CNN_. Точно так же локальные признаки из ветви _CNN_ постепенно возвращаются к эмбеддингам трансформера, чтобы обогатить информацией о локальных деталей его ветвь. 
`Conformer` состоит из стержневого модуля, двух ветвей, элементов _FCU_ для их соединения и двух классификаторов (слой _fc_) для каждой из ветвей.
Стержневой модуль представляет собой свертку _7 × 7_ с шагом 2, слоем _MaxPool_ _3 × 3_ с шагом 2. Каждая ветвь состоит из _N_ (например, 12) повторяющихся блоков свертки или трансформера. Такая параллельная структура подразумевает, что ветви _CNN_ и трансформера могут соответственно сохранять в максимальной степени локальные особенности и глобальные представления. 

Авторы предлагают три вариации `Conformer` : `Conformer-S`, `Conformer-B` и `Conformer-Ti`. Архитектура `Conformer-S` представлена в Таблице 1. По сравнению с `Conformer-S`, `Conformer-Ti` уменьшает количество каналов ветви _CNN_ на 1/4, а `Conformer-B` увеличивает количество каналов в ветви _CNN_ и размерность эмбеддингов в ветви трансформера в 1,5 раза.

<div align="center">
  <img src="https://github.com/kontik-pk/diplom/blob/main/nets/transformers/illustrations/Architecture_of_Conformer_S.png" width="500" />
</div></br>
<div align="center">
  <figcaption><i> Таблица 1. Архитектура Conformer-S</i> </figcaption>
</div></br>

Для каждой из ветвей есть свой классификатор. Во время обучения используется _CrossEntropyLoss_, чтобы контроллировать обучение каждой из ветвей отдельно. 
Эмпирически установлено, что важность функций потерь одинакова. Во время валидации и тестирования, выходные данные двух классификаторов просто суммируются как результаты предсказания.

При объединении двух ветвей нужно учитывать, что карты признаков из _CNN_ имеют размерность _C × H × W_ (_C, H, W_ - количество каналов, высота и ширина соответственно), в то время как размерность выхода блока ветви трансформера - _(K + 1) × E_, где _K_, _1_ и _E_ соответственно представляют количество участков изображения, маркер класса и размер эмбеддинга. Для передачи выхода _CNN_ в ветвь трансформера, карты признаков сначала должны пройти через свертку _1 × 1_, чтобы выровнять номера каналов встраиваемых эмбеддингов. Затем используется модуль понижающей дискретизации для завершения выравнивания пространственного измерения. При передаче выхода трансформера ветви _CNN_, эмбеддинги проходят через слой повышающей дискретизации для выравнивания пространственного масштаба, и через свертку _1 × 1_ для возвращения исходного количества каналов.

`Conformer` обучали на датасете  `ImageNet` на протяжении 300 эпох с использованием оптимизатора _AdamW_, размер батча составлял 1024 изображений, начальная скорость обучения 0.001 и понижалась на протяжении обучения с помощью планировщика. Результаты на валидационной выборке показали, что модель `Conformer` превосходит существующие архитектуры _CNN_ и трансформеров. `Conformer-S` (c 37.7M параметров) показал результат на 4.1% лучше, чем `ResNet-152` (c 60.2M параметров), и на  1.6% лучше, чем `DeiT-B` (c 86.6M параметров). Кроме того, `Conformer` не только показал лучшие результаты, но и обучался быстрее.

Авторами статьи [Conformer: Local Features Coupling Global Representations for Visual Recognition](https://arxiv.org/pdf/2105.03889v1.pdf) также был проведен эксперимент: модели `Conformer`, `DeiT`, `ResNet-50` и `ResNet-152`, предобученные на датасете `ImageNet`, тестировались на выборке с изображениями разного размера (от 112×112 до 448×448 пикселей). Исследование показало (Рисунок 3), что `Conformer` показывает лучшие результаты на всех выборках, причем разница в результатах на каждой их выборок минимальна среди всех используемых моделей. Это доказывает, что модель `Conformer` лучше всего подстраивается под данные разного размера, то есть в некоторой степени обладает лучшей способностью к генерализации.

<div align="center">
  <img src="https://github.com/kontik-pk/diplom/blob/main/nets/transformers/illustrations/Generalization_capability.png" width="500" />
</div></br>
<div align="center">
  <figcaption>Рис. 3. <i> Сравнение масштабной инвариантности. Модели обучаются на изображениях с разрешением 224 × 224 и тестируются на разных разрешениях изображений </i> </figcaption>
</div></br>

В настоящее время модель `Conformer-B` находится на 113 месте соревнования по классификации изображений на датасете `ImageNet`и показывает качество 84.1%. 

# Источники

- [ResT: An Efficient Transformer for Visual Recognition](https://arxiv.org/pdf/2105.13677v5.pdf)
- [Conformer: Local Features Coupling Global Representations for Visual Recognition](https://arxiv.org/pdf/2105.03889v1.pdf)
- [An image is worth 16X16 words: transformers for image recognition at scale](https://arxiv.org/pdf/2010.11929v2.pdf)