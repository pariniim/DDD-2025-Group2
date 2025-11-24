# DDD-2025-Group2
Oleksandra Drapushko, Andrii Ioffe, Maria Mititelu, Isabel Parini

## Human Perception on Alien Related Content
![Word Cloud from Comments](./DATAALIENS/analysis_outputs/comment_wordcloud.png "Word Cloud from Comments")

![Final Visualization 1](./Final_Visualization_05.jpg "Partial Visualization")

### Abstract
Our project focuses on comments for videos about aliens published on TikTok from September 15, 2020 to November 21, 2025, in order to understand how extraterrestrials and their sightings are perceived by users of this social medium.

Data was scraped with an APIFY tool. The sentiment analysis to evaluate whether a comment was positive, negative, or neutral was also done using an APIFY tool. 

### Protocol Diagram 
```mermaid
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
```

### What topic does the project address?
We investigated how aliens are perceived across comments to some videos published on TikTok. 

### What data have you considered?
* Comments on TikTok videos related to aliens published between September 15, 2020 and November 21, 2025.

* An automated sentiment analysis was then used to evaluate the comments to understand wether they were positive, negative, or neutral. 

### Dataset 
[Link to datasets](./DATAALIENS/)

### What does the visualisation show?
* The 10 most relevant videos in terms of engagement; 
* Word clouds in UFO shapes that illustrate the most recurring words in the comments for each video;
* Results of the sentiment analyis showing whether the comments tended to be positive, neutral, or negative; 
* The timeline from 2020 to 2025 shows when the selected videos were posted;
* The spikes indicate the quantity of posts in a certain time period and show an increase after the first half of the current year (2025).
