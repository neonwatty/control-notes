import sys
import re
from bs4 import BeautifulSoup

# extract filename
filename = str(sys.argv[1])
soup = BeautifulSoup(open(filename),"lxml")

# page_title becomes the page title and series becomes the series name (duh!)
foo = soup.find_all('h1')[0]
page_title = foo.next_element
foo = soup.find_all('h2')[0]
series = foo.next_element

# name will become the filename: eg, name.html and name.ipynb
name = soup.html.head.title.string

# change title to page_title
soup.html.head.title.string = page_title

# assign series URL
if 'differentiation' in series.lower():
    series_url = '3_Automatic_differentiation'
    
if 'zero' in series.lower():
    series_url = '5_Zero_order_methods'
        
if 'first' in series.lower():
    series_url = '6_First_order_methods'    
    
if 'second' in series.lower():
    series_url = '7_Second_order_methods'    
    
if 'linear' in series.lower() and 'regression' in series.lower():
    series_url = '8_Linear_regression'   
    
if 'linear' in series.lower() and 'unsupervised' in series.lower():
    series_url = '11_Linear_unsupervised_learning'
    
if 'reinforcement' in series.lower() and 'foundation' in series.lower():
    series_url = '18_Reinforcement_Learning_Foundations'                           
  
if 'multiclass' in series.lower() and 'linear' in series.lower():
    series_url = '10_Linear_multiclass_classification'    

if 'classification' in series.lower() and 'linear' in series.lower():
    series_url = '9_Linear_twoclass_classification'  

if 'nonlinear' in series.lower() and 'introduction' in series.lower():
    series_url = '12_Nonlinear_intro'   
    
if 'layer' in series.lower() and 'perceptron' in series.lower():
    series_url = '13_Multilayer_perceptrons'
    
if 'convolution' in series.lower() and 'network' in series.lower():
    series_url = '14_Convolutional_networks'   

if 'recurrent' in series.lower() and 'network' in series.lower():
    series_url = '15_Recurrent_Networks'  
    
if 'selection' in series.lower() and 'engin' in series.lower():
    series_url = '9_Feature_engineer_select' 
    
if 'nonlinear' in series.lower() and 'feature' in series.lower():
    series_url = '10_Nonlinear_intro'  
    
if 'principle' in series.lower() and 'learning' in series.lower():
    series_url = '11_Feature_learning'  
    
if 'element' in series.lower() and 'algebra' in series.lower():
    series_url = '16_Computational_linear_algebra'           
    
          
         

# This script adds navigation bar + sharing logos + title
script_1 = '''
<!-- uncomment to add back menu
<div style="text-align:center !important; padding-top:58px;">

				<a href="../../../index.html" style="font-family: inherit; font-weight: 200; letter-spacing: 1.5px; color: #222; font-size: 97%;">HOME</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="../../../about.html" style="font-family: inherit; font-weight: 200; letter-spacing: 1.5px; color: #222; font-size: 97%;">ABOUT</a>


</div> -->

<br><br><br>

<!-- share buttons -->
<div style="width: 53%; margin:auto;">

	<div id="1" style="width: 75%; float:left;">
		<span style="color:black; font-family:'lato', sans-serif; font-size: 18px;">code</span>
		<div style="width: 95px; border-bottom: solid 1px; border-color:black;">

			<div class="logo-share"></div>
			<div class="logo-share"></div>

			<div class="logo-share">
				<!-- github -->
				<a target="_blank" href="https://github.com/jermwatt/mlrefined/blob/gh-pages/blog_posts/'''+series_url+'''/'''+ name+'''.ipynb">
					<img src="../../html/pics/github.png" width=28 height=28 onmouseover="this.src='../../html/pics/github_filled.png';" onmouseout="this.src='../../html/pics/github.png';">
				</a>
			</div>
		</div>
	</div>

	<div id="2" style="width: 25%; float:left;">
		<span style="color:black; font-family:'lato', sans-serif; font-size: 18px;">share</span>
		<div style="width: 210px; border-bottom: solid 1px; border-color:black;">

			<div class="logo-share"></div>
			<div class="logo-share"></div>

			<div class="logo-share">
				<!-- linkedin -->
				<a target="_blank" href="https://www.linkedin.com/cws/share?url=https%3A%2F%2Fjermwatt.github.io%2Fmlrefined%2Fblog_posts%2F'''+series_url+'''%2F'''+ name+'''.html">
					<img src="../../html/pics/linkedin.png" width=28 height=28 onmouseover="this.src='../../html/pics/linkedin_filled.png';" onmouseout="this.src='../../html/pics/linkedin.png';">
				</a>
			</div>

			<div class="logo-share"></div>

			<div class="logo-share">
				<!-- twitter -->
				<a target="_blank" href="https://twitter.com/intent/tweet?ref_src=twsrc%5Etfw&tw_p=tweetbutton&url=https%3A%2F%2Fjermwatt.github.io%2Fmlrefined%2Fblog_posts%2F'''+series_url+'''%2F'''+ name+'''.html">
					<img src="../../html/pics/twitter.png" width=28 height=28 onmouseover="this.src='../../html/pics/twitter_filled.png';" onmouseout="this.src='../../html/pics/twitter.png';">
				</a>
			</div>

			<div class="logo-share"></div>

			<div class="logo-share">
				<!-- facebook -->
				<a target="_blank" href="https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fjermwatt.github.io%2Fmlrefined%2Fblog_posts%2F'''+series_url+'''%2F'''+ name+'''.html">
					<img src="../../html/pics/facebook.png" width=28 height=28 onmouseover="this.src='../../html/pics/facebook_filled.png';" onmouseout="this.src='../../html/pics/facebook.png';">
				</a>
			</div>
		</div>

	</div>
</div>

<br><br>

<div class="page-title" style="text-align: center !important;">
	<span style="color: #333; font-size: 40%; letter-spacing: 3px;">
		<span style="font-size: 80%;">&#x25BA; </span><a target="_blank" href="https://jermwatt.github.io/mlrefined/index.html" style="color: black; cursor: pointer; text-transform: uppercase; font-weight:bold;">'''+ series + '''</a>
	</span>
	<br><br>
	<mark style="padding: 0px; background-color: #f9f3c2;">'''+ page_title +'''</mark>
</div>
<br>'''

# parse script as BeautifulSoup object
html_1 = BeautifulSoup(script_1,'html.parser')

# insert it as the first element of the body tag, hence [0]
soup.body.insert(0, html_1)


# # This script adds comment section to the bottom of the page
# script_2 = '''
# <br><br><br><br><br><br>

# <!-- comment section -->
# <div id="disqus_thread" style="width:70%; height:auto; margin:auto;"></div>
# <script>
# (function() { // DON'T EDIT BELOW THIS LINE
# var d = document, s = d.createElement('script');
# s.src = 'https://mlrefined.disqus.com/embed.js';
# s.setAttribute('data-timestamp', +new Date());
# (d.head || d.body).appendChild(s);
# })();
# </script>
# <noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>
# '''

# # parse script as BeautifulSoup object
# html_2 = BeautifulSoup(script_2,'html.parser')

# # insert it as the last element of body tag, hence: -1
# soup.body.insert(-1, html_2)


# This script changes default LateX font to a prettier version
script_3 = '''

    <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
    	TeX: { equationNumbers: { autoNumber: "AMS" } },
        tex2jax: {
            inlineMath: [ ['$','$'], ["\\\(","\\\)"] ],
            displayMath: [ ['$$','$$'], ["\\\[","\\\]"] ],
            processEscapes: true,
            processEnvironments: true
        },
        // Center justify equations in code and markdown cells. Elsewhere
        // we use CSS to left justify single line equations in code cells.
        displayAlign: 'center',
        "HTML-CSS": {
            availableFonts: ["TeX"],
            preferredFont: "TeX",
            styles: {'.MathJax_Display': {"margin": 0}},
            linebreaks: { automatic: true }
        }
    });
    </script>

    <link href="../../html/CSS/custom.css" rel="stylesheet"/>

    <style>
        p {
            text-align: justify !important;
            text-justify: inter-word !important;
        }
    </style>

    '''
# parse script as BeautifulSoup object
html_3 = BeautifulSoup(script_3, 'html.parser')

# replace the old font with the new font
soup.head.find(text=re.compile(r'HTML-CSS')).parent.replace_with(html_3);


# you have to render soup again (for some reason) before you can search it
soup = BeautifulSoup(soup.renderContents(),"lxml")

# remove old title
soup.body.find_all('h1')[0].decompose()

# remove old series title
soup.body.find_all('h2')[0].decompose()

# remove code cells that contain the following message
# 'in the HTML version'
for cell in soup.body.find_all(text=re.compile('in the HTML version')):
	cell.parent.parent.parent.parent.decompose()


# finish by spiting out modified soup as html
with open(filename, "wt") as file:
    file.write(str(soup))

print('----------------')
print('Conversion done!')
print(' ')
print('   ¯\\_(ツ)_/¯')
print(' ')
print('----------------')
