# Hello!

This is lab5

# How dows it work?

As far as you might know, the task is count vehicles (or cars?) in image. So, we decided to use number of recognizable license plates on image as cars.  
Disadvantages:  
 - if it's obvious criminal vehicle, it have no number. **This case we consider as "basically out of bounds"**
 - the model won't detect the car if it's far away and lp is not recognizable. 

Advantages: 
 - According to general rules, every vehicle should have licenseplate number, so it's as much easy to detecet track and bike
 - We can give basic licenseplate recognition as feature

# How to run
You may see the example by  <a href ="https://colab.research.google.com/drive/1LvCnl8JoPbZ--EV2JLZben0hD01zm96K?usp=sharing"> this link </a>

# Dependencies
You may find dependencies in `install.sh` file

# Dockerfile
It is for deploy this staff as lp-recognizer web app, **do not run dockerfile**
