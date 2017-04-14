# Computing for Mental Health Project

Research in the field of visualization often leverages work by cognitive and perceptual psychology to provide the foundation for new lines of inquiry. Unfortunately, there is a disconnect between the tightly-controlled laboratory studies being referenced and the application of visualization tools in practice. This sometimes results in altered performance or unexpected analytical behaviors which are difficult to explain.

Indeed, the various fields that study human behavior have observed this problem for ages: laboratory psychology, clinical psychology, sociology, and social work (just to name a few) each take very different approaches to observing and explaining various behavioral phenomena. In collaboration with researchers and practitioners in the field of human services at the Justice Resource Institute (JRI), we are working to understand how our historically separate disciplines might better be able to support one another [1]:

 - Leveraging what visualization research has learned about how to support complex reasoning, we work to co-create tools that can make an impact on the availability and efficacy of community-based mental health resources.
 - In tandem, this collaboration provides opportunities for visualization researchers to learn complementary techniques for assessing and modulating human behavior with an emphasis on individual well-being and the well-being of society.

For more information on specific components of this project, as well as the broader topic of visualization for social justice, please contact us.

[1] Crouser, R.J. and Crouser, M.R. "Mind the Gap: the Importance of Pluralistic Discourse in Computing for Mental Health." To appear at the 2016 Workshop on Computing for Mental Health at the ACM SIGCHI Conference on Human-Computer Interaction.

# Data Curation & Wrangling

Raw data was collected from the online DSM-V website: http://dsm.psychiatryonline.org/doi/10.1176/appi.books.9780890425596.dsm02 using the BeautifulSoup web-scraping package in Python. The output is a .txt file listing an array of codes with the same, corresponding chapter and symptom (i.e. ['(F70)', '(F71)', '(F72)', '(F73)']|Chapter: 1|b'Onset of intellectual and adaptive deficits during the developmental period.') Refer to "DSM-V Web-Scrape.py". 

Through series of consultation with clinicians from the JRI, we categorized the data into 29 different identifying categories that would best identify a diagnosis using binary notation. If a diagnosis fit a category, we assigned a '1'. If not, a '0'. A significant part of the process was done manually because it required context-based human interpretation of the symptoms. Otherwise, Python and R was used to wrangle the data. Refer to "160720 Pivot.csv" for the wrangled data. The legitimacy and accuracy of our binary classifications are currently in the process of cross-validation with clinicians via crowd-sourcing. 

# Machine Learning Analysis 

The framework of the decision tree to guide clinicians to make a better diagnosis was created in Python using scikit-learn, an open-source package for data mining and analysis, and Graphviz, an open-source software for graph visualization. 

Analysis using random forests helped determine whether the splits of the branches made by the decision tree were biased. The random forest module available in scikit-learn, matplotlib (a Python plotting library), and Seaborn (a Python statistical data visualization library) was made. Refer to "Random Forest.py". 

Analysis using heat maps highlight significant patterns of the position and frequency within each category with respect to varying numbers of trees and features. Refer to "HeatMap - tree variation.py" & "HeatMap Creator, varying number of features.py". 

# Clinician Itnerface 

The decision tree was formatted into JSON to create the front-end, interactive visualization of the decision tree in D3.js. Refer to "Tree to JSON.py" or "Tree to JSON.ipynb. This is still an on-going process. 

# Researcher's Interface
 To be updated. 
