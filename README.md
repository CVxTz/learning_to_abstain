### Know What You Don’t Know: Getting Reliable Confidence Scores When Unsure of a Prediction

Code for: https://towardsdatascience.com/know-what-you-dont-know-getting-reliable-confidence-scores-when-unsure-of-a-prediction-882f2f146726

Exposure to out of sample examples as a way to get more meaningful prediction
scores.

Softmax predicion scores are often used as a confidence score in a multi-class
classification setting. In this post, we are going to show that softmax scores
can be meaningless when doing regular empirical risk minimization by gradient
descent. We are also going to apply the method presented in [Deep Anomaly
Detection with Outlier Exposure](https://arxiv.org/abs/1812.04606) to mitigate
this problem and add more meaning to the softmax score.

Discriminative classifiers (models that try to estimate P(y|x) from data) tend
to be overconfident in their predictions, even if the input sample looks nothing
like anything they have seen in the training phase. This makes it so that the
output scores of such models cannot be reliably used as a confidence score since
the model is often confident where it should not be.

### Example :

In this synthetic example, we have one big cluster of class zero and another one
for class one, plus two smaller groups of points of outliers that were not
present in the training set.

![](https://cdn-images-1.medium.com/max/800/1*6PYjL96Azr1lI9pvNEMoOQ.jpeg)


If we apply a regular classifier to this we get something like this :

![](https://cdn-images-1.medium.com/max/800/1*dSyDRCbsBCjNor2dgQkTuA.jpeg)

We see that the classifier is overly confident everywhere, even the outlier
samples are classified with a very high score. **The confidence score is
displayed using the heat-map**.

This is what makes it a bad idea to directly use the softmax scores as
confidence scores, if a classifier is confident everywhere without having seen
any evidence to support it in the training then it probably means that the
confidence scores are wrong.

However, if we use the approach presented in [Deep Anomaly Detection with
Outlier Exposure](https://arxiv.org/abs/1812.04606) we can achieve much more
reasonable Softmax scores :

![](https://cdn-images-1.medium.com/max/800/1*bg3-SJWh1fhDbFTd6_F5LQ.jpeg)


This score map is much more reasonable and is useful to see where the model is
rightly confident and where it is not. The outlier region has a very low
confidence ~0.5 ( Equivalent to no confidence at all in a two-class setting).

### Description of the Approach

The idea presented in [Deep Anomaly Detection with Outlier
Exposure](https://arxiv.org/abs/1812.04606) is to use external data that is
mostly different from your training/test data and force the model to predict the
uniform distribution on this external data.

For example, if you are trying to build a classifier that predicts cat vs dog in
images, you can get a bunch of bear and shark images and force the model to
predict \[0.5, 0.5\] on those images.

### Data And Model

We will use the 102 Flower as the in-distribution dataset and a subset of the
OpenImage dataset as an out-of-distribution dataset. In the paper referenced in
the introduction, they show that training on one set of out-of-distribution
samples generalizes well to other sets that are out-of-distribution.

We use MobilenetV2 as our classification architecture and initialize the weights
with Imagenet.

    get_model_classification(
        input_shape=(
    , 
    , 3),
        weights=
    ,
        n_classes=102,
    ):
        inputs = Input(input_shape)
        base_model = MobileNetV2(
            include_top=
    , input_shape=input_shape, weights=weights
        )

        x = base_model(inputs)
        x = Dropout(0.5)(x)
        out1 = GlobalMaxPooling2D()(x)
        out2 = GlobalAveragePooling2D()(x)
        out = Concatenate(axis=-1)([out1, out2])
        out = Dropout(0.5)(out)
        out = Dense(n_classes, activation=
    )(out)
        model = Model(inputs, out)
        model.compile(
            optimizer=Adam(0.0001), loss=categorical_crossentropy, metrics=[
    ]
        )

        
    model

We will use a generator to load the images from the hard drive batch by batch.
In the baseline we only load the in-distribution images while in the Anomaly
exposure model we load half the batch from in-distribution images with their
correct label and the other half from out-of-distribution images with a uniform
objective => :

    target = [1 / n_label 
    _ 
    range(n_label)]

### Results

Both training configurations get a little higher than 90% accuracy on
in-distribution samples. We choose to predict “Don’t Know” if the softmax score
is lower than 0.15 and thus abstain from making a class prediction.

Now let us see how each model behaves!

#### Regular training :

You can run the web application by doing :

    streamlit run main.py

![](https://cdn-images-1.medium.com/max/600/1*-hN4n4o2yefb-Hxq25V8cw.png)

![](https://cdn-images-1.medium.com/max/600/1*oqqdPPFzqA0Yk7lkVTeVsw.png)

<span class="figcaption_hack">First **bird** image from Unsplash / second **flower** image from 102 Flowers
dataset</span>

We trained a flower classification and then tried to classify a picture of a
bird, the model predicts the class cyclamen with a pretty high confidence
“**0.72**” which is not what we want.

#### Anomaly exposure :

You can run the web application by doing :

    streamlit run ood_main.py

![](https://cdn-images-1.medium.com/max/600/1*8sBYaC8z90FZAmj0K4V-6g.png)

![](https://cdn-images-1.medium.com/max/600/1*kCP4X0tZr77V2529d5fR-Q.png)

<span class="figcaption_hack">First **bird** image from Unsplash / second **flower** image from 102 Flowers
dataset</span>

Much better ! The bird picture is classified as “Don’t Know” since the maximum
softmax score is 0.01. The prediction of the in-distribution flower image
(right) stayed the same between the two models.

Next are predictions on images from Wikipedia, the first one (left) is of a
flower that is represented in the 102 flowers dataset and the second one is of a
carnivorous plant that does not exist in the distribution. The Anomaly exposed
model behaved as expected, the in-distribution image is correctly classified
while the model abstains from making a class prediction for the carnivorous
plant.

![](https://cdn-images-1.medium.com/max/600/1*BWXTdRL25e0p7MxgJp2sfg.png)

![](https://cdn-images-1.medium.com/max/800/1*1RsZhKRt6wPyWJlDShe01A.png)

<span class="figcaption_hack">Modified First **flower** image By Aftabbanoori — Own work, CC BY-SA 4.0,
[https://commons.wikimedia.org/w/index.php?curid=41960730](https://commons.wikimedia.org/w/index.php?curid=41960730)
/ Modified Second **plant** image:
[https://commons.wikimedia.org/wiki/File:Venus_Flytrap_showing_trigger_hairs.jpg](https://commons.wikimedia.org/wiki/File:Venus_Flytrap_showing_trigger_hairs.jpg)</span>

### Conclusion

In this post, we showed how flawed the softmax scores can be as a measure of
confidence. We then applied a method based on outlier exposure to fix this
problem and to get more meaningful confidence scores. This allows us to reliably
abstain from making a prediction if needed, which is a crucial feature in many
business/research applications where it is better to not make any prediction at
all than make an obviously wrong one.

#### References:

[Deep Anomaly Detection with Outlier Exposure](https://arxiv.org/abs/1812.04606)
