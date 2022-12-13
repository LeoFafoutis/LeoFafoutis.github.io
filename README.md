# Rating players in the Overwatch League
### LeoFafoutis

## Introduction
Overwatch 1 was a 6v6 first person shooter developed by Activision Blizzard in 2016. The game focuses on two teams trying to capture objectives or accomplish tasks.
The three roles in the game (tank, damage, and support) offer a variety of playstyles within the game.

A competitive scene grew over the years and a professional league began in 2018 named The Overwatch League (OWL). There are now 20 teams all competing in a regular
season ending with a playoff bracket of some sort. 

In this tutorial, we want to create a way to rank individual players based on their statistics for each role. We can 
then use this “Player Impact Rating” to predict the outcome of the playoffs and compare it to the actual results. This will allow us to see how well statistics can 
predict a team’s placement. To gain a better idea of what a Player Impact Rating would be, the Overwatch Leagues’ analyst attempted to create a Player Impact Rating 
(PIR) to rank player performance across roles (Read More). In short, it uses a variety of factors to compare players across roles to see which player, by statistics, 
is the best. For the purposes of this tutorial, we will compare players in their respective roles, since we may want to rating healing higher to support players. 

However, in order to create this PIR, we need a way to rate which statistics are going to be most important in our algorithm. To figure out which stats are important, 
we can use efficiency metrics and compare them to both the OWL’s data as well as team placement.

We will also learn how to view data from a CSV, extract the statistics we need, and analyze it to better understand the data science pipeline.

## Data Collection

To begin, we will be using Python along with pandas, numpy, matplotlib, seaborn, and more to handle the data. First, we need to import the correct libraries into our 
file as shown below.
  
```
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
```



This site was built using [Read More](https://overwatchleague.com/en-us/news/23051823/introducing-player-impact-rating)
