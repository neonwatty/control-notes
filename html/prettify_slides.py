import sys
import re
from bs4 import BeautifulSoup

# extract filename
filename = str(sys.argv[1])
soup = BeautifulSoup(open(filename),"lxml")

# name will become the filename: eg, name.html and name.ipynb
name = soup.html.head.title.string

# remove code cells that contain the following message
# 'in the HTML version'
for cell in soup.body.find_all(text=re.compile('in the HTML version')):
	cell.parent.parent.parent.parent.decompose()

# remove in/out 
for thought_leader in soup.findAll("div", {"class": "prompt output_prompt"}):
	if 'Out' in thought_leader.text:
		thought_leader.decompose()

for thought_leader in soup.findAll("div", {"class": "prompt input_prompt"}):
	if 'In' in thought_leader.text:
		thought_leader.decompose()


# finish by spiting out modified soup as html
with open(filename, "wt") as file:
    file.write(str(soup))

print('----------------')
print('Conversion done!')
print(' ')
print('   ¯\\_(ツ)_/¯')
print(' ')
print('----------------')
