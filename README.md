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

May 15
Over the last week, I moved from mostly researching Statcast concepts into figuring out how to actually structure the project technically. A major focus was understanding the limitations of the public dataset and determining how to approximate variables that MLB does not directly provide. I spent time looking into whether outfielder starting positions were publicly available and found that while exact play-by-play positioning is mostly proprietary, resources like Baseball Savant provide average outfielder positioning data that can be used as a reasonable approximation. I also worked on connecting raw tracking variables such as hc_x, hc_y, launch_speed, and launch_angle to more meaningful defensive metrics related to catch difficulty. In addition, I planned out how the dataset would be converted into a machine learning problem by defining a binary outcome variable (catch vs. non-catch) and identifying which engineered features would likely contribute most to prediction accuracy. Most of this week was spent on understanding how to bridge the gap between raw Statcast data and a simplified version of Catch Probability rather than directly training a model immediately.






