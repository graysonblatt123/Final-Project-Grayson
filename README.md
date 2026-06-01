# Final-Project-Grayson


I know this isn't a lot of work yet, and this write up was last minute. Due to the speech and debate tournament, and frequent tests, I've only just started to keep a journal.

1. I did extensive research to understand how Statcast calculates Catch Probability (April 16)\
https://www.mlb.com/news/catch-probability-updated-to-include-direction-c232532408\
https://www.mlb.com/news/statcast-introduces-catch-probability-for-2017-c217802340\

Statcast defines Catch Probability as the likelihood that a fielder successfully records an out on a batted ball. Through research, I identified the primary factors influencing this metric:

Distance Covered (how far the fielder must travel)\
Time to Catch Opportunity (Hang Time)\
Direction of Movement (forward, backward, lateral)\
Proximity to the fence https://www.mlb.com/news/catch-probability-updated-to-account-for-walls-c269814542


These variables reflect both physical difficulty and reaction constraints for players\

2. Dataset Selection & Constraints April (20-23)\

A Statcast-derived dataset was selected containing batted ball tracking data. I'll aim for one season of data, tracking:

hc_x, hc_y → ball landing coordinates\
launch_speed, launch_angle\
hit_distance\
fielder_7/8/9 (outfielders)\
events (out vs hit)\


https://www.kaggle.com/datasets/s903124/mlb-statcast-data?

## April 29
Read through MLB Statcast articles to understand what Catch Probability measures and what factors go into it, especially distance, hang time, and direction.

## April 30
Looked more closely at how Statcast tracks batted balls and how raw tracking data is turned into useful metrics for evaluating defense.

## May 1
Explored the Kaggle dataset and went through the columns to understand what each variable represents and which ones might be useful.

## May 2
Decided to frame the project as predicting whether a ball is caught or not using a classification model.

## May 3
Started cleaning the dataset by keeping only balls put into play and removing rows with missing values.

## May 4
Set up the basic structure for the model, including defining the outcome variable and thinking about which inputs to use.

## May 5
Added a few simple features based on the data (like estimated distance and general ball location) and finalized the plan to test a basic model like logistic regression or random forest.

## May 15

Over the last week, I moved from mostly researching Statcast concepts into figuring out how to actually structure the project technically. A major focus was understanding the limitations of the public dataset and determining how to approximate variables that MLB does not directly provide. I spent time looking into whether outfielder starting positions were publicly available and found that while exact play-by-play positioning is mostly proprietary, resources like Baseball Savant provide average outfielder positioning data that can be used as a reasonable approximation. I also worked on connecting raw tracking variables such as hc_x, hc_y, launch_speed, and launch_angle to more meaningful defensive metrics related to catch difficulty. In addition, I planned out how the dataset would be converted into a machine learning problem by defining a binary outcome variable (catch vs. non-catch) and identifying which engineered features would likely contribute most to prediction accuracy. Most of this week was spent on understanding how to bridge the gap between raw Statcast data and a simplified version of Catch Probability rather than directly training a model immediately.

## May 22
Spent time learning about the differences between logistic regression and XGBoost to decide which model would be more appropriate for predicting catch outcomes. Watched introductory videos explaining how each model works and took notes on their strengths, weaknesses, and ideal use cases.

Logistic Regression Notes:

Works well for binary classification problems (such as catch vs. no catch).
Produces probabilities, making predictions easier to interpret.
Assumes a relatively simple relationship between variables and outcomes.
Easier to understand and explain, which may help when presenting the project.

XGBoost Notes:

Uses decision trees and boosting to improve prediction accuracy.
Better at modeling complex, non-linear relationships in data.
Can handle feature interactions automatically without manually engineering as many variables.
More powerful for prediction, but harder to interpret and requires more tuning.

Project Overview | May 31st

This project creates a simplified version of MLB Statcast’s Catch Probability system using public baseball data and machine learning. The goal is to predict whether an outfielder will catch a ball based on details of the play.

The project compares two models: logistic regression and XGBoost, with XGBoost used as the main model because it can better recognize patterns in baseball data.

## Key Features

The model uses several important features to make predictions:

- Distance Needed → how far the fielder has to run
- Hang Time → how long the ball stays in the air
- Direction → where the fielder has to move
- Near Wall → whether the play happens near the warning track

The model also includes:

Whether the ball is a line drive
A time-to-distance ratio, which measures how much time the fielder has to cover distance
Single-Play Prediction

The program can also predict one specific play. A user enters information about the play, including:

Distance run
Hang time
Direction of the ball
Whether it is near the wall
Whether it is a line drive

The model then gives a catch probability (for example, 75%) and a difficulty rating similar to MLB Statcast.

## Model Performance

The model was evaluated using ROC curves, calibration curves, and feature importance.

XGBoost outperformed logistic regression, achieving an AUC score of 0.97, compared to 0.91 for logistic regression. This means the model was highly effective at separating catches from non-catches.

The calibration curve showed that predicted probabilities closely matched real outcomes, meaning the model’s predictions were generally reliable.

Feature importance analysis found that hang time (opportunity time) was the strongest predictor of a catch, followed by launch angle and distance needed. This suggests that reaction time plays the biggest role in catch difficulty.

## Conclusion

This project shows that public baseball data can be used to build a model that estimates how difficult a catch is. Even though it does not have MLB’s private tracking data, it can still make useful predictions for individual plays and overall defensive difficulty.

## Extra info:
I used 2021 data, as strikeouts since then have increased by 22 percent. Hopefully a stronger offense in 2021 meant that the range of flyballs hit were more diverse

I added the launch angle as a feature, which led to a significant increase in accuracy of the data. Instead of only classifying a hit ball as a line drive or a fly ball in a binary manner, I added this feature, as not all fly balls or linedrives are equally easy to read off the bat, depending on launch angle (This presumably makes catch probability more accurate) 

Here's a sample of a difficult catch and it's stats to plug into the model, to see if it passes the eye test:
(Use baseball savant to find your own catches)
https://baseballsavant.mlb.com/sporty-videos?playId=da11f02b-1f51-3ba9-a7c9-dc43cfe4db92 

Pete Crow-Armstrong's diving catch against the Cardinals

Launch angle: 14 degrees

Bbdist (ball distance): 333 ft

Fielder hit to: 8

Near warning track: no

Line Drive: yes


# I was going to add a joblib reload function to streamline singular predictions, but I ran out of time. Thanks for the year!


