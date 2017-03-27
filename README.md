# airbnKEY

![Demo](product_demo.gif)


## Note on Confidentiality
This project was conducted in collaboration with Loftium (http://www.loftium.com/).  Due to the confidentiality of some of the data and modeling methodology, only a subset of information is provided on this repository.

If you are a recruiter who requires full access to the project or product demo for evaluation purposes, please send me an inquiry at john.mk.kim@gmail.com. 


## Project Description
Home affordability continues to be a major challenge in many metropolitan areas.  With housing prices continuing to outpace wage growth, it has become increasingly difficult for prospective home buyers to fulfill the American dream.

Fortunately, homeowners now have the option to Airbnb spare rooms in their new homes to help pay for the mortgage.  

This project aims to estimate the dollar benefit a homeowner can expect from a given home.

## Repository Structure

- app : content for flask web app
- src : python source files
- walkthroughs : jupyter notebook walkthroughs for regression modeling and natural language processing


## CRISP-DM Workflow

### Data Understanding
#### Data Sources:
- Partner data
- Airbnb listing data

#### Obtaining the data:
- Web-scraping
- Available APIs

### Data Exploration
- Statistics on dataset
- Visualizations 

### Data Preparation
- Cleaning data for null values
- Managing outliers
- Feature engineering
- Joining datasets

### Modeling
This project will employ an ensemble approach using the following models:

#### Gradient Boosting Regression
Use gradient boosting to predict the the Airbnb income for a spare room for a given home.

#### NLP
Use unsupervised natural language processing methods to extract features and/or topics from unstructured data on airbnb listing pages.

## Evaluation
To determine the optimal regression model, this project used Mean Squared Error as the key evaluation metric.

Methods used:
- Cross-validation
- Bootstrapping
- Grid-search

## Deployment
The final product is a basic web app using flask. 

The app allows a user to look up expected daily income from a room based on various features. 

See `product_demo.gif` for a demo of the web app. 