# DDD-2025-Group2
Oleksandra Drapushko, Andrii Ioffe, Maria Mititelu, Isabel Parini

## Human Perception on Alien Related Content

![Final Visualization 1](./Final_Visualization_05.jpg "Partial Visualization")

![Final Visualization 2](./Final_Visualization_2_05.png "Final Visualization")

### Abstract
Our project focuses on a sentiment analysis about how media about aliens - in our case videos on TikTok - are perceived by users. 

In order to do so, we scraped data using Apify (https://apify.com/) and collected comments on videos about extraterrestrials published on Tiktok, focusing on a time range going from September 15, 2020 to November 21, 2025. 

Our data presents the most frequent words used in the comments for the 10 most popular videos of our original dataseas as a word cloud (one word cloud per video), as well as the result of a sentiment analysis which shows how many comments are positive, negative, or neutral in percentages. 

### Protocol Diagram 

![Protocol Diagram](./protocol_diagram.jpg "Protocolo Diagram")

### What topic does the project address?
We chose to explore the perception of aliens by focusing on comments about videos on TikTok related to extraterrestrials. 

### What data have you considered?
Comments on TikTok videos related to aliens published between September 15, 2020 and November 21, 2025.

### Dataset 
flowchart TB
    A["Topic:<br>Alien sightings"] --> B["How people react to the content about aliens on social media(TikTok)"] & n10["What are the main trends in the alien content on tiktok"]
    B --> C{"Collect data"}
    C --> D["TikTok posts (2021-2025)"] & n2["TikTok Posts comments"] & n6["List of reported UFO sightings"]
    D --> F{"Organize & code"}
    F --> G["Grouped dataset"]
    G --> H{"Analysis"}
    H --> I["Sentiment analysis and visualization (avarage comment sentiment over time)"] & J["Publication rates and spikes analysis, visualization"] & n3["monthly engagment visualization"] & n4["Word Clouds"] & n5["most common UFO shape analysis?"]
    I --> N{"Interpretation"}
    J --> N
    N --> O["Insights:<br><br>"] & n7["Combined visualization"]
    n2 --> F
    n3 --> N
    n4 --> N
    n5 --> N
    n6 --> n5
    O --> n8["The most common UFO shape is a disc"] & n9["People most often see UFOs at the end of the year"] & n12["Most of comments are neutral with the trend on enlarging the range of emations"] & n13["Most of videos on the topic is AI generated"] & n14["people connect aliens with politics and hot topic of the time(epstein files)"]
    n10 --> C

    A@{ shape: rect}
    D@{ shape: cyl}
    n2@{ shape: cyl}

### What does the visualisation show?
* The 10 most relevant videos; 
* Word clouds in UFO shapes that illustrate the most recurring words in the comments for each video;
* Results of the sentiment analyis in percentages that show how many comments were either positive, neutral, or negative; 
* The timeline shows when the videos were posted;
* The spikes indicate the quantity of posts in a certain time period and shows an increase after the first half of the current year (2025).
