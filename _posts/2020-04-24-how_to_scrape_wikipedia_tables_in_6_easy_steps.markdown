---
layout: post
title:      "How to scrape Wikipedia tables in 6 easy steps."
date:       2020-04-24 01:19:56 -0400
permalink:  how_to_scrape_wikipedia_tables_in_6_easy_steps
---

### What is scraping?

According to Wikipedia : 
>' Web scraping, web harvesting, or web data extraction is data scraping used for extracting data from websites. Web scraping software may access the World Wide Web directly using the Hypertext Transfer Protocol, or through a web browser. While web scraping can be done manually by a software user, the term typically refers to automated processes implemented using a bot or web crawler. It is a form of copying, in which specific data is gathered and copied from the web, typically into a central local database or spreadsheet, for later retrieval or analysis'.

 I will be scraping a Wikipedia page to gather information about all the movies released in 1999 and here is the link for that search: https://en.wikipedia.org/wiki/List_of_American_films_of_1999
 
 ![Wikipedia page](https://i.imgur.com/pdJKaGb.png)
 
 We need a table with titles, genres, director and actor names. Almost all tables in Wikipedia have the same tree structure so don't be surprised if you will be able to scrape some other tables following the same methodology.
 
 Import Requests, Beautiful Soup and Pandas:
    `import requests`
    `from bs4 import BeautifulSoup`
    ` import pandas as pd`

Using Requests and Beautiful Soup we extract all the info from the HTML, the result is going to be pretty but messy.
```
html = urlopen(url) 
soup = BeautifulSoup(html, 'html.parser')
website_url = requests.get(url).text
soup = BeautifulSoup(website_url, 'lxml')
```
To be able to find HTML elements we have to inspect element on the web page. 
Right click on the table:
 ![Click on 'Inspect'](https://i.imgur.com/zd17VGk.png)
 
There is an actual element 'table' in HTML:
 
 ![Table element](https://i.imgur.com/MgcFC6v.png)
 
Here is the code:
    
		``My_table = soup.find('table',{'class':'wikitable sortable'})``
 
 The HTML element for row is <tr>
 ![Row element](https://i.imgur.com/FTyrBdY.png)
 
 Here is the code to get all rows:
   
	 ``rows = My_table.find_all('tr')``

The HTML element for cell in the row is <td>
 ![Cell element]([https://i.imgur.com/ZkD1s9a.png)
 
 Here is the code to get all cells in a row:
   
	 ``cells = row.find_all('td')``
 
 The information we need sits inside each cell in 'a' element marked with `<a ` with attribute `title`.

![Title](https://i.imgur.com/bWuirSs.png)

Here is the code to get the information from the cell:
Counting starts with 0 so the title of the movie sits in the 0 cell.
```title = cells[0] ```
Director's name sits in the 1st cell.
```director = cells[1]```
Actors' names sit in 2nd cell.
```actor = cells[2]```
You can find genre information in the 3rd cell.
```genre = cells[3]```

We know how to extract information from each cell. The following code will allow you to pull a whole table using a for loop and dump it into a data frame.
```
titles = []
directors = []
actors = []
genres = []
rows = My_table.find_all('tr')
for row in rows:
     cells = row.find_all('td')
     if len(cells)>=1:
          title = cells[0]
          titles.append(title.text)
          director = cells[1]
          directors.append(director.text)
          actor = cells[2]
          actors.append(actor.text)
          genre = cells[3]
          genres.append(genre.text)
          index = [*range(1,len(titles)+1,1)]
          df = pd.DataFrame({'Titles': titles,'Directors' : directors, 'Actors': actors, 'Genres':    genres, }, index = index   )
```

Et voila!
![Result Table](https://i.imgur.com/kp749uh.png)

Good luck scraping!
 





