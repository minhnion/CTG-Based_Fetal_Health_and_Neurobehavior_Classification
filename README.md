This project aims to develop a machine learning model that classifies fetal conditions based on Cardiotocography (CTG) signals. The model performs two main classification tasks:

Fetal Health Classification: Predicting the overall clinical state of the fetus into three categories:

Normal

Suspect

Pathologic

Neurobehavioral State Classification: Identifying various neurobehavioral states of the fetus, such as:

Calm sleep

REM sleep

Calm vigilance

Active vigilance

Shift pattern

Accelerative/decelerative pattern

Decelerative pattern

Largely decelerative

Flat-sinusoidal (pathological state)

Suspect pattern

Main Input Features
The key input features extracted from CTG signals include, but are not limited to:

Baseline and Variability Metrics:

Baseline value (manually annotated and computed by systems like SisPorto),

Accelerations,

Decelerations (light, severe, prolonged),

Fetal movement,

Uterine contractions.

Short-Term and Long-Term Variability:

ASTV (percentage of time with abnormal short-term variability),

mSTV (mean value of short-term variability),

ALTV (percentage of time with abnormal long-term variability),

mLTV (mean value of long-term variability).

Histogram-Based Features:

Width, Minimum, Maximum, Mode, Mean, Median, Variance, and Tendency of the histogram derived from the FHR signal.

Project Objective
The objective of this project is to build a model that can effectively interpret the key CTG signals and simultaneously predict:

The overall fetal health condition (Normal, Suspect, or Pathologic),

And the neurobehavioral state (represented by labels from calm sleep through suspect pattern).

Future enhancements may include detailed evaluation of the model's performance, further tuning of the hyperparameters, and eventual integration into a clinical decision support system.

