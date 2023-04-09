---
title: "[논문] Segment Anything"
subtitle: 👩🏻‍💻
categories : [paper]
tags: [Instance, Segmentation]
author: Summer
show_author_profile: true
key: paper
permalink: /paper/segment-anything/
---



# Abstract

We introduce the Segment Anything (SA) project: a new task, model, and dataset for image segmentation.
Using our efficient model in a data collection loop, we built the largest segmentation dataset to date (by far), with over 1 billion masks on 11M licensed and privacy respecting images.
The model is designed and trained to be promptable, so it can transfer zero-shot to new image distributions and tasks.
We evaluate its capabilities on numerous tasks and find that its zero-shot performance is impressive – often competitive with or even superior to prior fully supervised results.
We are releasing the Segment Anything Model (SAM) and cor- responding dataset (SA-1B) of 1B masks and 11M images at [https://segment-anything.com](https://segment-anything.com/) to foster research into foun- dation models for computer vision.

# 1. Introduction

Large language models pre-trained on web-scale datasets are revolutionizing NLP with strong zero-shot and few-shot generalization [10].
These “foundation models” [8] can generalize to tasks and data distributions beyond those seen during training.
This capability is often implemented with prompt engineering in which hand-crafted text is used to prompt the language model to generate a valid textual re- sponse for the task at hand.
When scaled and trained with abundant text corpora from the web, these models’ zero and few-shot performance compares surprisingly well to (even matching in some cases) fine-tuned models [10, 21].
Empir- ical trends show this behavior improving with model scale, dataset size, and total training compute [56, 10, 21, 51].

Foundation models have also been explored in computer vision, albeit to a lesser extent.
Perhaps the most promi- nent illustration aligns paired text and images from the web.
For example, CLIP [82] and ALIGN [55] use contrastive learning to train text and image encoders that align the two modalities.
Once trained, engineered text prompts enable zero-shot generalization to novel visual concepts and data distributions.
Such encoders also compose effectively with other modules to enable downstream tasks, such as image generation (e.g., DALL·E [83]).
While much progress has been made on vision and language encoders, computer vi- sion includes a wide range of problems beyond this scope, and for many of these, abundant training data does not exist.

In this work, our goal is to build a foundation model for image segmentation.
That is, we seek to develop a prompt- able model and pre-train it on a broad dataset using a task that enables powerful generalization.
With this model, we aim to solve a range of downstream segmentation problems on new data distributions using prompt engineering.

The success of this plan hinges on three components: task, model, and data.
To develop them, we address the following questions about image segmentation:

1. What task will enable zero-shot generalization?
2. What is the corresponding model architecture?
3. What data can power this task and model?

These questions are entangled and require a comprehen- sive solution.
We start by defining a promptable segmenta- tion task that is general enough to provide a powerful pre- training objective and to enable a wide range of downstream applications.
This task requires a model that supports flex- ible prompting and can output segmentation masks in real- time when prompted to allow for interactive use.
To train our model, we need a diverse, large-scale source of data.
Unfortunately, there is no web-scale data source for seg- mentation;to address this, we build a “data engine”, i.e., we iterate between using our efficient model to assist in data collection and using the newly collected data to improve the model.
We introduce each interconnected component next, followed by the dataset we created and the experiments that demonstrate the effectiveness of our approach.

### Task (§2).

In NLP and more recently computer vision, foundation models are a promising development that can perform zero-shot and few-shot learning for new datasets and tasks often by using “prompting” techniques.
Inspired by this line of work, we propose the promptable segmen- tation task, where the goal is to return a valid segmenta- tion mask given any segmentation prompt (see Fig. 1a).
A prompt simply specifies what to segment in an image, e.g., a prompt can include spatial or text information identifying an object.
The requirement of a valid output mask means that even when a prompt is ambiguous and could refer to multiple objects (for example, a point on a shirt may in- dicate either the shirt or the person wearing it), the output should be a reasonablemask for at least one of those ob- jects.
We use the promptable segmentation task as both a pre-training objective and to solve general downstream seg- mentation tasks via prompt engineering.

### Model (§3).

The promptable segmentation task and the goal of real-world use impose constraints on the model architec- ture.
In particular, the model must support flexible prompts, needs to compute masks in amortized real-time to allow in- teractive use, and must be ambiguity-aware.
Surprisingly, we find that a simple design satisfies all three constraints: a powerful image encoder computes an image embedding, a prompt encoder embeds prompts, and then the two infor- mation sources are combined in a lightweight mask decoder that predicts segmentation masks.
We refer to this model as the Segment Anything Model, or SAM (see Fig. 1b).
By separating SAM into an image encoder and a fast prompt encoder / mask decoder, the same image embedding can be reused (and its cost amortized) with different prompts.
Given an image embedding, the prompt encoder and mask decoder predict a mask from a prompt in ∼50ms in a web browser.
We focus on point, box, and mask prompts, and also present initial results with free-form text prompts.
To make SAM ambiguity-aware, we design it to predict mul- tiple masks for a single prompt allowing SAM to naturally handle ambiguity, such as the shirt vs. person example.

### Data engine (§4).

To achieve strong generalization to new data distributions, we found it necessary to train SAM on a large and diverse set of masks, beyond any segmenta- tion dataset that already exists.
While a typical approach for foundation models is to obtain data online [82], masks are not naturally abundant and thus we need an alternative strategy.
Our solution is to build a “data engine”, i.e., we co-develop our model with model-in-the-loop dataset an- notation (see Fig. 1c).
Our data engine has three stages: assisted-manual, semi-automatic, and fully automatic.
In the first stage, SAM assists annotators in annotating masks, similar to a classic interactive segmentation setup.
In the second stage, SAM can automatically generate masks for a subset of objects by prompting it with likely object lo- cations and annotators focus on annotating the remaining objects, helping increase mask diversity.
In the final stage, we prompt SAM with a regular grid of foreground points, yielding on average ∼100 high-quality masks per image.

### Dataset (§5).

Our final dataset, SA-1B, includes more than 1B masks from 11M licensed and privacy-preserving im- ages (see Fig. 2).
SA-1B, collected fully automatically us- ing the final stage of our data engine, has 400× more masks than any existing segmentation dataset [66, 44, 117, 60], and as we verify extensively, the masks are of high qualityand diversity.
Beyond its use in training SAM to be robust and general, we hope SA-1B becomes a valuable resource for research aiming to build new foundation models.

### Responsible AI (§6).

We study and report on potential fair- ness concerns and biases when using SA-1B and SAM.
Im- ages in SA-1B span a geographically and economically di- verse set of countries and we found that SAM performs sim- ilarly across different groups of people.
Together, we hope this will make our work more equitable for real-world use cases.
We provide model and dataset cards in the appendix.

### Experiments (§7).

We extensively evaluate SAM.
First, us- ing a diverse new suite of 23 segmentation datasets, we find that SAM produces high-quality masks from a single fore- ground point, often only slightly below that of the manu- ally annotated ground truth.
Second, we find consistently strong quantitative and qualitative results on a variety of downstream tasks under a zero-shot transfer protocol using prompt engineering, including edge detection, object pro- posal generation, instance segmentation, and a preliminary exploration of text-to-maskprediction.
These results sug- gest that SAM can be used out-of-the-box with prompt en- gineering to solve a variety of tasks involving object and image distributions beyond SAM’s training data.
Neverthe- less, room for improvement remains, as we discuss in §8.

### Release.

We are releasing the SA-1B dataset for research purposes and making SAM available under a permissive open license (Apache 2.0) at [https://segment-anything.com](https://segment-anything.com/).
We also showcase SAM’s capabilities with an online demo.

# 2. Segment Anything Task

We take inspiration from NLP, where the next token pre- diction task is used for foundation model pre-training and to solve diverse downstream tasks via prompt engineer- ing [10].
To build a foundation model for segmentation, we aim to define a task with analogous capabilities.

### Task.

We start by translating the idea of a prompt from NLP to segmentation, where a prompt can be a set of foreground / background points, a rough box or mask, free-form text, or, in general, any information indicating what to segment inan image.
The promptable segmentation task, then, is to return a valid segmentation mask given any prompt.
The re- quirement of a “valid” mask simply means that even when a prompt is ambiguous and could refer to multiple objects (e.g., recall the shirt vs. person example, and see Fig. 3), the output should be a reasonable maskfor at least one of those objects.
This requirement is similar to expecting a lan- guage model to output a coherent response to an ambiguous prompt.
We choose this task because it leads to a natural pre-training algorithm and a general method for zero-shot transfer to downstream segmentation tasks via prompting.

### Pre-training.

The promptable segmentation task suggests a natural pre-training algorithm that simulates a sequence of prompts (e.g., points, boxes, masks) for each training sam- ple and compares the model’s mask predictions against the ground truth.
We adapt this method from interactive seg- mentation [109, 70], although unlike interactive segmenta- tion whose aim is to eventually predict a valid mask after enough user input, our aim is to always predict a valid mask for any prompt even when theprompt is ambiguous.
This ensures that a pre-trained model is effective in use cases that involve ambiguity, including automatic annotation as re- quired by our data engine §4.
We note that performing well at this task is challenging and requires specialized modeling and training loss choices, which we discuss in §3.

### Zero-shot transfer.

Intuitively, our pre-training task en- dows the model with the ability to respond appropriately to any prompt at inference time, and thus downstream tasks can be solved by engineering appropriate prompts.
For ex- ample, if one has a bounding box detector for cats, cat in- stance segmentation can be solved by providing the detec- tor’s box output as a prompt to our model.
In general, a wide array of practical segmentation tasks can be cast as prompt- ing.
In addition to automatic dataset labeling, we explore five diverse example tasks in our experiments in §7.

### Related tasks.

Segmentation is a broad field: there's in- teractive segmentation [57, 109], edge detection [3], su- per pixelization [85], object proposal generation [2], fore- ground segmentation [94], semantic segmentation [90], in- stance segmentation [66], panoptic segmentation [59], etc.
The goal of our promptable segmentation task is to produce a broadly capable model that can adapt to many (though not all) existing and new segmentation tasks via prompt engineering.
This capability is a form of task generaliza- tion [26].
Note that this is different than previous work on multi-task segmentation systems.
In a multi-task system, a single model performs a fixed set of tasks, e.g., joint seman- tic, instance, and panoptic segmentation [114, 19, 54], but the training and test tasks are the same.
An important dis- tinction in our work is that a model trained for promptable segmentation can perform a new, different task at inference time by acting as a component in a larger system, e.g., to perform instance segmentation, a promptable segmentation model is combined withan existing object detector.

### Discussion.

Prompting and composition are powerful tools that enable a single model to be used in extensible ways, po- tentially to accomplish tasks unknown at the time of model design.
This approach is analogous to how other founda- tion models are used, e.g., how CLIP [82] is the text-image alignment component of the DALL·E [83] image generation system.
We anticipate that composable system design, pow- ered by techniques such as prompt engineering, will enable a wider variety of applications than systems trained specif- ically for a fixed set of tasks.
It's also interesting to com- pare promptable and interactive segmentation through the lens of composition: while interactive segmentation mod- els are designed with human users in mind, a model trained for promptable segmentation can also be composed into a larger algorithmic system as we will demonstrate.

# 3. Segment Anything Model
![Untitled](https://user-images.githubusercontent.com/121393261/230776828-4064ba02-ae98-4086-b147-8d6e028b8248.png)
>> Figure 4: Segment Anything Model (SAM) overview. A heavyweight image encoder outputs an image embedding that can then be efficiently queried by a variety of input prompts to produce object masks at amortized real-time speed. For ambiguous prompts corresponding to more than one object, SAM can output multiple valid masks and associated confidence scores.

Figure 4: Segment Anything Model (SAM) overview. A heavyweight image encoder outputs an image embedding that can then be efficiently queried by a variety of input prompts to produce object masks at amortized real-time speed. For ambiguous prompts corresponding to more than one object, SAM can output multiple valid masks and associated confidence scores.

We next describe the Segment Anything Model (SAM) for promptable segmentation.
SAM has three components, illustrated in Fig. 4: an image encoder, a flexible prompt encoder, and a fast mask decoder.
We build on Transformer vision models [14, 33, 20, 62] with specific tradeoffs for (amortized) real-time performance.
We describe these com- ponents at a high-level here, with details in §A.

### Image encoder.

Motivated by scalability and powerful pre- training methods, we use an MAE [47] pre-trained Vision Transformer (ViT) [33] minimally adapted to process high resolution inputs [62].
The image encoder runs once per image and can be applied prior to prompting the model.

### Prompt encoder.

We consider two sets of prompts: sparse (points, boxes, text) and dense (masks).
We represent points and boxes by positional encodings [95] summed with learned embeddings for each prompt type and free-form text with an off-the-shelf text encoder from CLIP [82].
Dense prompts (i.e., masks) are embedded using convolutions and summed element-wise with the image embedding.

### Mask decoder.

The mask decoder efficiently maps the im- age embedding, prompt embeddings, and an output token to a mask.
This design, inspired by [14, 20], employs a modification of a Transformer decoder block [103] followed by a dynamic mask prediction head.
Our modified decoder block uses prompt self-attention and cross-attention in two directions (prompt-to-image embedding and vice-versa) to update all embeddings.
After running two blocks, we up- sample the image embedding and an MLP maps the output token to a dynamic linear classifier, which then computes the mask foreground probability at each image location.

### Resolving ambiguity.

With one output, the model will av- erage multiple valid masks if given an ambiguous prompt.
To address this, we modify the model to predict multiple output masks for a single prompt (see Fig. 3).
We found 3 mask outputs is sufficient to address most common cases (nested masks are often at most three deep: whole, part, and subpart).
During training, we backprop only the minimum loss [15, 45, 64] over masks.
To rank masks, the model pre- dicts a confidence score (i.e., estimated IoU) for each mask.

### Efficiency.

The overall model design is largely motivated by efficiency.
Given a precomputed image embedding, the prompt encoder and mask decoder run in a web browser, on CPU, in ∼50ms.
This runtime performance enables seam- less, real-time interactive prompting of our model.

### Losses and training.

We supervise mask prediction with the linear combination of focal loss [65] and dice loss [73] used in [14].
We train for the promptable segmentation task using a mixture of geometric prompts (for text prompts see §7.5).
Following [92, 37], we simulate an interactive setup by randomly sampling prompts in 11 rounds per mask, al- lowing SAM to integrate seamlessly into our data engine.

# 4. Segment Anything Data Engine

As segmentation masks are not abundant on the inter- net, we built a data engine to enable the collection of our 1.1B mask dataset, SA-1B.
The data engine has three stages: (1) a model-assisted manual annotation stage, (2) a semi-automatic stage with a mix of automatically predicted masks and model-assisted annotation, and (3) a fully auto- matic stage inwhich our model generates masks without annotator input.
We go into details of each next.

### Assisted-manual stage.

In the first stage, resembling clas- sic interactive segmentation, a team of professional annota- tors labeled masks by clicking foreground / background ob- ject points using a browser-based interactive segmentation tool powered by SAM.
Masks could be refined using pixel- precise “brush” and “eraser” tools.
Our model-assisted an- notation runs in real-time directly inside a browser (using precomputed image embeddings) enabling a truly interac- tive experience.
We did not impose semantic constraints for labeling objects, and annotators freely labeled both “stuff” and “things” [1].
We suggested annotators label objects they could name or describe, but did not collect these names or descriptions.
Annotators were asked to label objects in order of prominence and were encouraged to proceed to the next image once a mask took over 30 seconds to annotate.

At the start of this stage, SAM was trained using com- mon public segmentation datasets.
After sufficient data an- notation, SAM was retrained using only newly annotated masks.
As more masks were collected, the image encoder was scaled from ViT-B to ViT-H and other architectural de- tails evolved;in total we retrained our model 6 times.
Av- erage annotation time per mask decreased from 34 to 14 seconds as the model improved.
We note that 14 seconds is 6.5× faster than mask annotation for COCO [66] and only 2× slower than bounding-box labeling with extreme points [76, 71].
As SAM improved, the average number of masks per image increased from 20 to 44 masks.
Overall, we collected 4.3M masks from 120k images in this stage.

### Semi-automatic stage.

In this stage, we aimed to increase the diversity of masks in order to improve our model’s ability to segment anything.
To focus annotators on less prominent objects, we first automatically detected confident masks.
Then we presented annotators with images prefilled with these masks and asked them to annotate any additional unannotated objects.
To detect confident masks, we trained a bounding box detector [84] on all first stage masks using a generic “object” category.
During this stage we collected an additional 5.9M masks in 180k images (for a total of 10.2M masks).
As in the first stage, we periodically retrained our model on newly collected data (5 times).
Average annota- tion time per mask went back up to 34 seconds (excluding the automatic masks) as these objects were more challeng- ing to label.
The average number of masks per image went from 44 to 72 masks (including the automatic masks).

### Fully automatic stage.

In the final stage, annotation was fully automatic.
This was feasible due to two major en- hancements to our model.
First, at the start of this stage, we had collected enough masks to greatly improve the model, including the diverse masks from the previous stage.
Sec- ond, by this stage we had developed the ambiguity-aware model, which allowed us to predict valid masks even in am- biguous cases.
Specifically, we prompted the model with a 32×32 regular grid of points and for each point predicted a set of masks that may correspond to valid objects.
With the ambiguity-aware model, if a point lies on a part or sub- part, our model will return the subpart, part, and whole ob- ject.
The IoU prediction module of our model is used to se- lect confident masks;moreover, we identified and selected only stable masks (we consider a mask stable if threshold- ing the probability map at 0.5 − δ and 0.5 + δ results in similar masks).
Finally, after selecting the confident and stable masks, we applied non-maximal suppression (NMS) to filter duplicates.
To further improve the quality of smaller masks, we also processed multiple overlapping zoomed-in image crops.
For further details of this stage, see §B.
We applied fully automatic mask generation to all 11M images in our dataset, producing a total of 1.1B high-quality masks.
We describe and analyze the resulting dataset, SA-1B, next.

![Untitled 1](https://user-images.githubusercontent.com/121393261/230776825-a681c007-8f5d-4d01-ae29-fe1d40935136.png)


# 5. Segment Anything Dataset

Our dataset, SA-1B, consists of 11M diverse, high- resolution, licensed, and privacy protecting images and 1.1B high-quality segmentation masks collected with our data engine.
We compare SA-1B with existing datasets and analyze mask quality and properties.
We are releasing SA-1B to aid future development of foundation models for computer vision.
We note that SA-1B will be released un- der a favorable license agreement for certain research uses and with protections for researchers.

### Images.

We licensed a new set of 11M images from a provider that works directly with photographers.
These im- ages are high resolution (3300×4950 pixels on average), and the resulting data size can present accessibility and stor- age challenges.
Therefore, we are releasing downsampled images with their shortest side set to 1500 pixels.
Even af- ter downsampling, our images are significantly higher reso- lution than many existing vision datasets (e.g., COCO [66] images are ∼480×640 pixels).
Note that most models today operate on much lower resolution inputs.
Faces and vehicle license plates have been blurred in the released images.

### Masks.

Our data engine produced 1.1B masks, 99.1% of which were generated fully automatically.
Therefore, the quality of the automatic masks is centrally important.
We compare them directly to professional annotations and look at how various mask properties compare to prominent seg- mentation datasets.
Our main conclusion, as borne out in the analysis below and the experiments in §7, is that our automatic masks are high quality and effective for training models.
Motivated by these findings, SA-1B only includes automatically generated masks.

### Mask quality.

To estimate mask quality, we randomly sam- pled 500 images (∼50k masks) and asked our professional annotators to improve the quality of all masks in these im- ages.
Annotators did so using our model and pixel-precise “brush” and “eraser” editing tools.
This procedure resulted in pairs of automatically predicted and professionally cor- rected masks.
We computed IoU between each pair and found that 94% of pairs have greater than 90% IoU (and 97% of pairs have greater than 75% IoU).
For comparison, prior work estimates inter-annotator consistency at 85-91% IoU [44, 60].
Our experiments in §7 confirm by human rat- ings that mask quality is high relative to a variety of datasets and that training our model on automatic masks is nearly as good as using all masks produced by the data engine.

### Mask properties.

In Fig. 5 we plot the spatial distribution of object centers in SA-1B compared to the largest existing segmentation datasets.
Common photographer biases are present in all datasets.
We observe that SA-1B has greater coverage of image corners compared to LVIS v1 [44] and ADE20K [117], the two most similarly distributed datasets, while COCO [66] and Open Images V5 [60] have a more prominent center bias.
In Fig. 6 (legend) we compare these datasets by size.
SA-1B has 11× more images and 400× more masks than the second largest, Open Images.
On av- erage, it has 36× more masks per image than Open Images.
The closest dataset in this respect, ADE20K, still has 3.5× fewer masks per image.
Fig. 6 (left) plots the masks-per- image distribution.
Next, we look at image-relative mask size (square root of the mask area divided by image area) in Fig. 6 (middle).
As expected, since our dataset has more masks per image, it also tends to include a greater percent- age of small and medium relative-size masks.
Finally, to analyze shape complexity, we look at mask concavity (1 minus mask area divided by area of mask’s convex hull) in Fig. 6 (right).
Since shape complexity is correlated with mask size, we control for the datasets’ mask size distribu- tions by first performing stratified sampling from binned mask sizes.
We observe that the concavity distribution of our masks is broadly similar to that of other datasets.

# 6. Segment Anything RAI Anaysis

We next perform a Responsible AI (RAI) analysis of our work by investigating potential fairness concerns and bi- ases when using SA-1B and SAM.
We focus on the geo- graphic and income distribution of SA-1B and fairness of SAM across protected attributes of people.
We also provide dataset, data annotation, and model cards in §F.

### Geographic and income representation.

We infer the country images were photographed in using standard meth- ods (see §C).
In Fig. 7 we visualize the per-country image counts in SA-1B (left) and the 50 countries with the most images (right).
We note that the top-three countries are from different parts of the world.
Next, in Table 1 we com- pare the geographic and income representation of SA-1B, COCO [66], and Open Images [60].
SA-1B has a substan- tially higher percentage of images in Europe and Asia & Oceania as well as in middle income countries.
All datasets underrepresent Africa as well as low income countries.
We note that in SA-1B, all regions, including Africa, have at least 28 million masks, 10× more than the total number of masks of any previous dataset.
Finally, we observe that the average number of masks per image (not shown) is fairly consistent across region and income (94-108 per image).

### Fairness in segmenting people.

We investigate potential fairness concerns across perceived gender presentation, per- ceived age group, and perceived skin tone by measuring the performance discrepancy of SAM between groups.
We use the More Inclusive Annotations for People (MIAP) [87] dataset for gender presentation and age and a proprietary dataset for skin tone (see §C).
Our evaluation uses simu- lated interactive segmentation with random sampling of 1 and 3 points (see §D).
Table 2 (top left) shows results for perceived gender presentation.
We note that females have been shown to be underrepresented in detection and seg- mentation datasets [115], but observe that SAM performs similarly across groups.
We repeat the analysis for per- ceived age in Table 2 (bottom left), noting that those who are perceived to be younger and older have been shown to be underrepresented in large-scale datasets [110].
SAM per- forms best on those who are perceived older (although the confidence interval is large).
Finally, we repeat the anal- ysis for perceived skin tone in Table 2 (right), noting that those with lighter apparent skin tones have been shown to be overrepresented and those with darker skin tones under- represented in large-scale datasets [110].
As MIAP does not contain perceived skin tone annotations, we use a pro- prietary dataset that contains annotations for the perceived Fitzpatrick skin type [36], which ranges from 1 (lightest skin tone) to 6 (darkest skin tone).
While the means vary somewhat, we do not find a significant difference across groups.
We believe our findings stem from the nature of the task, and acknowledge biases may arise when SAM is used as a component in larger systems.
Finally, in §C we extend the analysis to segmenting clothing where we find an indication of bias across perceived gender presentation.

# 7. Zero-Shot Transfer Experiments

In this section, we present zero-shot transfer experiments with SAM, the Segment Anything Model.
We consider five tasks, four of which differ significantly from the promptable segmentation task used to train SAM.
These experiments evaluate SAM on datasets and tasks that were not seen during training (our usage of “zero-shot transfer” follows its usage in CLIP [82]).
The datasets may include novel image distributions, such as underwater or ego-centric images (e.g. Fig. 8) that, to our knowledge, do not appear in SA-1B.

Our experiments begin by testing the core goal of promptable segmentation: producing a valid mask from any prompt.
We emphasize the challenging scenario of a single foreground point prompt, since it is more likely to be am- biguous than other more specific prompts.
Next, we present a sequence of experiments that traverse low, mid, and high- level image understanding and roughly parallel the histori- cal development of the field.
Specifically, we prompt SAM to (1) perform edge detection, (2) segment everything, i.e. object proposal generation, (3) segment detected objects, i.e. instance segmentation, and (4), as a proof-of-concept, to segmentobjects from free-form text.
These four tasks dif- fer significantly from the promptable segmentation task that SAM was trained on and are implemented via prompt engi- neering.
Our experiments conclude with an ablation study.

### Implementation.

Unless otherwise specified: (1) SAM uses an MAE [47] pre-trained ViT-H [33] image encoder and (2) SAM was trained on SA-1B, noting that this dataset includes only automatically generated masks from the final stageof our data engine.
For all other model and training details, such as hyperparameters, refer to §A.

## 7.1. Zero-Shot Single Point Vaild Mask Evaluation

### Task.

We evaluate segmenting an object from a single fore- ground point.
This task is ill-posed as one point can refer to multiple objects.
Ground truth masks in most datasets do not enumerate all possible masks, which can make au- tomatic metrics unreliable.
Therefore, we supplement the standard mIoU metric (i.e., the mean of all IoUs between predicted and ground truth masks) with a human study in which annotators rate mask quality from 1 (nonsense) to 10 (pixel-perfect).
See §D.1, §E, and §G for additional details.
By default, we sample points from the “center” of ground truth masks (at a maximal value of the mask’s interior dis- tance transform), following the standard evaluation proto- col in interactive segmentation [92].
Since SAM is capable of predicting multiple masks, we evaluate only the model’s most confident mask by default.
The baselines are all single-mask methods.
We compare mainly to RITM [92], a strong interactive segmenter that performs best on our benchmark compared to other strong baselines [67, 18].

### Datasets.

We use a newly compiled suite of 23 datasets with diverse image distributions.
Fig. 8 lists the datasets and shows a sample from each one (see appendix Table 7 for more details).
We use all 23 datasets for mIoU evaluation.
For the human study, we use the subset listed in Fig. 9b (due to the resource requirements of such studies).
This subset includes both datasets for which SAM outperforms and underperforms RITM according to automatic metrics.

### Results.

First, we look at automatic evaluation on the full suite of 23 datasets using mIoU.
We compare per-dataset results in Fig. 9a against RITM.
SAM yields higher re- sults on 16 of the 23 datasets, by as much as ∼47 IoU.
We also present an “oracle” result, in which the most relevant of SAM’s 3 masks is selected by comparing them to the ground truth, rather than selecting the most confident mask.
This reveals the impact of ambiguity on automatic evalu- ation.
In particular, with the oracle to perform ambiguity resolution, SAM outperforms RITM on all datasets.

Results of the human study are presented in Fig. 9b.
Er- ror bars are 95% confidence intervals for mean mask rat- ings (all differences are significant; see §E for details).
We observe that the annotators consistently rate the quality of SAM’s masks substantially higher than the strongest base- line, RITM.
An ablated, “ambiguity-unaware” version of SAM with a single output mask has consistently lower rat- ings, though still higher than RITM.
SAM's mean ratings fall between 7 and 9, which corresponds to the qualitative rating guideline: “A high score (7-9): The object is identi- fiable and errors are small and rare (e.g., missing a small, heavily obscured disconnected component, ...).”
These re- sults indicate that SAM has learned to segment valid masks from a single point.
Note that for datasets like DRAM and IBD, where SAM is worse on automatic metrics, it receives consistently higher ratings in the human study.

Fig. 9c shows additional baselines, SimpleClick [67] and FocalClick [18], which obtain lower single point perfor- mance than RITM and SAM.
As the number of points in- creases from 1 to 9, we observe that the gap between meth- ods decreases.
This is expected as the task becomes easier;also, SAM is not optimized for the very high IoU regime.
Finally, in Fig. 9d we replace the default center point sam- pling with random point sampling.
We observe that the gap between SAM and the baselines grows and SAM is able to achieve comparable results under either sampling method.

## 7.2. Zero-Shot Edge Detection

### Approach.

We evaluate SAM on the classic low-level task of edge detection using BSDS500 [72, 3].
We use a sim- plified version of our automatic mask generation pipeline.
Specifically, we prompt SAM with a 16×16 regular grid of foreground points resulting in 768 predicted masks (3 per point).
Redundant masks are removed by NMS.
Then, edge maps are computed using Sobel filtering of unthresholded mask probability maps and standard lightweight postpro- cessing, including edge NMS (see §D.2 for details).

### Results.

We visualize representative edge maps in Fig. 10 (see Fig. 15 for more).
Qualitatively, we observe that even though SAM was not trained for edge detection, it produces reasonable edge maps.
Compared to the ground truth, SAM predicts more edges, including sensible ones that are not an- notated in BSDS500.
This bias is reflected quantitatively in Table 3: recall at 50% precision (R50) is high, at the cost of precision.
SAM naturally lags behind state-of-the-art meth- ods that learn the biases of BSDS500, i.e., which edges to suppress.
Nevertheless, SAM performs well compared to pioneering deep learning methods such as HED [108] (also trained on BSDS500) and significantly better than prior, though admittedly outdated, zero-shot transfer methods.

## 7.3. Zero-Shot Object Proposals

### Approach.

Next, we evaluate SAM on the mid-level task of object proposal generation [2, 102].
This task has played an important role in object detection research, serving as an intermediate step in pioneering systems (e.g., [102, 41, 84]).
To generate object proposals, we run a slightly modified version of our automatic mask generation pipeline and out- put the masks as proposals (see §D.3 for details).
We compute the standard average recall (AR) metric on LVIS v1 [44].
We focus on LVIS because its large number of categories presents a challenging test.
We compare to a strong baseline implemented as a ViTDet [62] detector (with cascade Mask R-CNN [48, 11] ViT-H).
We note that this “baseline” corresponds to the “Detector Masquerading as Proposal generator” (DMP) method [16] that was shown to game AR, making it a truly demanding comparison.

### Results.

In Table 4 we see unsurprisingly that using the detections from ViTDet-H as object proposals (i.e., the DMP method [16] that games AR) performs the best over- all.
However, SAM does remarkably well on several met- rics.
Notably, it outperforms ViTDet-H on medium and large objects, as well as rare and common objects.
In fact, SAM only underperforms ViTDet-H on small objects and frequent objects, where ViTDet-H can easily learn LVIS- specific annotation biases since it was trained on LVIS, un- like SAM.
We also compare against an ablated ambiguity- unaware version of SAM (“single out.”), which performs significantly worse than SAM on all AR metrics.

## 7.4. Zero-Shot Instance Segmentation

### Approach.

Moving to higher-level vision, we use SAM as the segmentation module of an instance segmenter.
The implementation is simple: we run a object detector (the ViTDet used before) and prompt SAM with its output boxes.
This illustrates composing SAM in a larger system.

### Results.

We compare the masks predicted by SAM and ViTDet on COCO and LVIS in Table 5.
Looking at the mask AP metric we observe gaps on both datasets, where SAM is reasonably close, though certainly behind ViTDet.
By visualizing outputs, we observed that SAM masks are often qualitatively better than those of ViTDet, with crisper boundaries (see §D.4 and Fig. 16).
To investigate this ob- servation, we conducted an additional human study asking annotators to rate the ViTDet masks and SAM masks on the 1 to 10 quality scale used before.
In Fig. 11 we observe that SAM consistently outperforms ViTDet in the human study.

We hypothesize that on COCO, where the mask AP gap is larger and the ground truth quality is relatively low (as borne out by the human study), ViTDet learns the specific biases of COCO masks.
SAM, being a zero-shot method, is unable to exploit these (generally undesirable) biases.
The LVIS dataset has higher quality ground truth, but there are still specific idiosyncrasies (e.g., masks do not contain holes, they are simple polygons by construction) and biases for modal vs. amodal masks.
Again, SAM is not trained to learn these biases, while ViTDet can exploit them.

## 7.5. Zero-Shot Text-to-Mask

### Approach.

Finally, we consider an even higher-level task: segmenting objects from free-form text.
This experiment is a proof-of-concept of SAM’s ability to process text prompts.
While we used the exact same SAM in all prior experiments, for this one SAM’s training procedure is mod- ified to make it text-aware, but in a way that does not require new text annotations.
Specifically, for each manually col- lected mask with area larger than 1002 we extract the CLIP image embedding.
Then, during training, we prompt SAM with the extracted CLIP image embeddings as its first in- teraction.
The key observation here is that because CLIP’s image embeddings are trained to align with its text embed- dings, we can train with image embeddings, but use text embeddings for inference.
That is, at inference time we run text through CLIP’s text encoder and then give the resulting text embedding as a prompt to SAM (see §D.5 for details).

### Results.

We show qualitative results in Fig. 12.
SAM can segment objects based on simple text prompts like “a wheel” as well as phrases like “beaver tooth grille”.
When SAM fails to pick the right object from a text prompt only, an additional point often fixes the prediction, similar to [31].

## 7.6. Ablations

We perform several ablations on our 23 dataset suite with the single center point prompt protocol.
Recall that a sin- gle point may be ambiguous and that ambiguity may not be represented in the ground truth, which contains only a single mask per point.
Since SAM is operating in a zero- shot transfer setting there can be systematic biases between SAM’s top-ranked mask vs. the masks resulting from data annotation guidelines.
We therefore additionally report the best mask with respect to the ground truth (“oracle”).

Fig. 13 (left) plots SAM’s performance when trained on cumulative data from the data engine stages.
We observe that each stage increases mIoU.
When training with all three stages, the automatic masks vastly outnumber the manual and semi-automatic masks.
To address this, we found that oversampling the manual and semi-automatic masks during training by 10× gave best results.
This setup complicates training.
We therefore tested a fourth setup that uses only the automatically generated masks.
With this data, SAM performs only marginally lower than using all data (∼0.5 mIoU).
Therefore, by default we use only the automatically generated masks to simplify the training setup.

In Fig. 13 (middle) we look at the impact of data volume.
The full SA-1B contains 11M images, which we uniformly subsample to 1M and 0.1M for this ablation.
At 0.1M im- ages, we observe a large mIoU decline under all settings.
However, with 1M images, about 10% of the full dataset, we observe results comparable to using the full dataset.
This data regime, which still includes approximately 100M masks, may be a practical setting for many use cases.

Finally, Fig. 13 (right) shows results with ViT-B, ViT-L, and ViT-H image encoders.
ViT-H improves substantially over ViT-B, but has only marginal gains over ViT-L.
Further image encoder scaling does not appear fruitful at this time.

# 8. Discussion

### Foundation models.

Pre-trained models have been adapted to downstream tasks since the early days of machine learn- ing [99].
This paradigm has become increasingly impor- tant in recent years with a growing emphasis on scale, and such models have recently been (re-)branded as “founda- tion models”: i.e. models that are “trained on broad data at scale and areadaptable to a wide range of downstream tasks” [8].
Our work correlates well with this definition, though we note that a foundation model for image segmen- tation is an inherently limited scope, since it represents an important, yet fractional, subset of computer vision.
We also contrast one aspect of our approach with [8], which emphasizes the role of self-supervised learning in founda- tion models.
While our model is initialized with a self- supervised technique (MAE [47]), the vast majority of its capabilities come from large-scale supervised training.
In cases where data engines can scale available annotations, like ours, supervised training provides an effective solution.

### Compositionality.

Pre-trained models can power new ca- pabilities even beyond ones imagined at the moment of training.
One prominent example is how CLIP [82] is used as a component in larger systems, such as DALL·E [83].
Our goal is to make this kind of composition straightfor- ward with SAM.
We aim to achieve this by requiring SAM to predict a valid mask for a wide range of segmentation prompts.
The effect is to create a reliable interface between SAM and other components.
For example, MCC [106] can easily use SAM to segment an object of interest and achieve strong generalization to unseen objects for 3D reconstruc- tion from a single RGB-D image.
In another example, SAM can be prompted with gaze points detected by a wearable device, enabling new applications.
Thanks to SAM’s abil- ity to generalize to new domains like ego-centric images, such systems work without need for additional training.

### Limitations.

While SAM performs well in general, it is not perfect.
It can miss fine structures, hallucinates small disconnected components at times, and does not produce boundaries as crisply as more computationally intensive methods that “zoom-in”, e.g.[18].
In general, we expect dedicated interactive segmentation methods to outperform SAM when many points are provided, e.g.[67].
Unlike these methods, SAM is designed for generality and breadth of use rather than high IoU interactive segmentation.
More- over, SAM can process prompts in real-time, but neverthe- less SAM’s overall performance is not real-time when using a heavy image encoder.
Our foray into the text-to-mask task is exploratory and not entirely robust, although we believe it can be improved with more effort.
While SAM can per- form many tasks, it is unclear how to design simple prompts that implement semantic and panoptic segmentation.
Fi- nally, there are domain-specific tools, such as [7], that we expect to outperform SAM in their respective domains.

### Conclusion.

The Segment Anything project is an attempt to lift image segmentation into the era of foundation models.
Our principal contributions are a new task (promptable seg- mentation), model (SAM), and dataset (SA-1B) that make this leap possible.
Whether SAM achieves the status of a foundation model remains to be seen by how it is used in the community, but regardless we expect the perspective of this work, the release of over 1B masks, and our promptable segmentation model will help pave the path ahead.

### Acknowledgments.

We would like to thank Aaron Ad- cock and Jitendra Malik for helpful discussion.
We thank Vaibhav Aggarwal and Yanghao Li for help with scal- ing the model.
We thank Cheng-Yang Fu, Jiabo Hu, and Robert Kuo for help with data annotation platform.
We thank Allen Goodman and Bram Wasti for help in optimiz- ing web-version of our model.
Finally, we thank Morteza Behrooz, Ashley Gabriel, Ahuva Goldstand, Sumanth Gur- ram, Somya Jain, Devansh Kukreja, Joshua Lane, Lilian Luong, Mallika Malhotra, William Ngan, Omkar Parkhi, Nikhil Raina, Dirk Rowe, Neil Sejoor, Vanessa Stark, Bala Varadarajan, and Zachary Winstrom for their help in mak- ing the demo, dataset viewer, and other assets and tooling.