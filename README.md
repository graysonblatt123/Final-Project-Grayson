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

April 29
Read through MLB Statcast articles to understand what Catch Probability measures and what factors go into it, especially distance, hang time, and direction.

April 30
Looked more closely at how Statcast tracks batted balls and how raw tracking data is turned into useful metrics for evaluating defense.

May 1
Explored the Kaggle dataset and went through the columns to understand what each variable represents and which ones might be useful.

May 2
Decided to frame the project as predicting whether a ball is caught or not using a classification model.

May 3
Started cleaning the dataset by keeping only balls put into play and removing rows with missing values.

May 4
Set up the basic structure for the model, including defining the outcome variable and thinking about which inputs to use.

May 5
Added a few simple features based on the data (like estimated distance and general ball location) and finalized the plan to test a basic model like logistic regression or random forest.








