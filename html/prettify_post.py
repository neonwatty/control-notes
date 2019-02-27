import sys
import re
from bs4 import BeautifulSoup

# extract filename
filename = str(sys.argv[1])
soup = BeautifulSoup(open(filename),"lxml")

# page_title becomes the page title and series becomes the series name (duh!)
foo = soup.find_all('h1')[0]
page_title = foo.next_element

# name will become the filename: eg, name.html and name.ipynb
name = soup.html.head.title.string

# change title to page_title
soup.html.head.title.string = page_title
 
	
# This script adds navigation bar + sharing logos + title
script_1 = '''

   <!-- Navigation -->
    <nav class="navbar navbar-default navbar-custom navbar-fixed-top">
        <div class="container-fluid">
            <!-- Brand and toggle get grouped for better mobile display -->
            <div class="navbar-header page-scroll">
                <button type="button" class="navbar-toggle" data-toggle="collapse" data-target="#bs-example-navbar-collapse-1">
                    <span class="sr-only">Toggle navigation</span>
                    Menu <i class="fa fa-bars"></i>
                </button>
                <a class="navbar-brand" href="https://dgsix.com">degree six</a>
            </div>

            <!-- Collect the nav links, forms, and other content for toggling -->
            <div class="collapse navbar-collapse" id="bs-example-navbar-collapse-1">
                <ul class="nav navbar-nav navbar-right">
                    <li>
                        <a href="../../index.html">Home</a>
                    </li>
                    <li>
                        <a href="../../subscribe.html">Subscribe</a>
                    </li>
                    <li>
                        <a href="https://dgsix.com/team">Authors</a>
                    </li>
                    <li>
                        <a href="https://dgsix.com/contact">Contact</a>
                    </li>
                </ul>
            </div>
            <!-- /.navbar-collapse -->
        </div>
        <!-- /.container -->
    </nav>


    <hr>

<div class="page-title" style="text-align: center !important;"> 
<!-- github
<div><a href="https://github.com/jermwatt/machine_learning_refined" style="text-decoration: none" target="_blank"><button class="btn-star">★ Our Project On GitHub</button></a></div>
-->
<div style="width: 100%; margin:auto;">

<div class="logo-share">
<!-- linkedin -->
<a href="https://www.linkedin.com/cws/share?url=https%3A%2F%2Fblog.dgsix.com%2F'''+ filename+ '''" target="_blank">
<img height="18" onmouseout="this.src='../../html/images/linkedin_off.png';" onmouseover="this.src='../../html/images/linkedin_on.png';" src="../../html/images/linkedin_off.png" width="18"/>
</a>
</div>


<div class="logo-share">
<!-- facebook -->
<a href="https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fblog.dgsix.com%2F'''+ filename+'''" target="_blank">
<img height="18" onmouseout="this.src='../../html/images/facebook_off.png';" onmouseover="this.src='../../html/images/facebook_on.png';" src="../../html/images/facebook_off.png" width="18"/>
</a>
</div>


<div class="logo-share">
<!-- twitter -->
<a href="https://twitter.com/intent/tweet?ref_src=twsrc%5Etfw&amp;tw_p=tweetbutton&amp;url=https%3A%2F%2Fblog.dgsix.com%2F'''+ filename+'''" target="_blank">
<img height="18" onmouseout="this.src='../../html/images/twitter_off.png';" onmouseover="this.src='../../html/images/twitter_on.png';" src="../../html/images/twitter_off.png" width="18"/>
</a>
</div>

<br><br>
	<div style="padding: 0px;">'''+ page_title +'''</div>
<svg width="80" height="15">
  <rect width="80" height="15" style="fill:#0085A1; stroke-width:0; stroke:rgb(0,0,0)" /> 
</svg>	
</div>




'''

# parse script as BeautifulSoup object
html_1 = BeautifulSoup(script_1,'html.parser')

# insert it as the first element of the body tag, hence [0]
soup.body.insert(0, html_1)


# # This script adds comment section to the bottom of the page
script_2 = '''
 
   <!-- Footer -->

 
 
 
 <br><br><br><br><br><br>

 <!-- comment section -->
 <div id="disqus_thread" style="width:70%; height:auto; margin:auto;"></div>
 <script>
 (function() { // DON'T EDIT BELOW THIS LINE
 var d = document, s = d.createElement('script');
 s.src = 'https://degreesix.disqus.com/embed.js';
 s.setAttribute('data-timestamp', +new Date());
 (d.head || d.body).appendChild(s);
 })();
 </script>
 <noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>
 '''

# # parse script as BeautifulSoup object
html_2 = BeautifulSoup(script_2,'html.parser')

# # insert it as the last element of body tag, hence: -1
soup.body.insert(len(soup.body.contents), html_2)


# This script changes default LateX font to a prettier version
script_3 = '''

	<meta property="og:title" content="'''+page_title+'''">
	<meta property="og:image" content="https://raw.githubusercontent.com/jermwatt/machine_learning_refined/gh-pages/html/pics/meta.png">
	<meta property="og:url" content="https://blog.dgsix.com/'''+filename+'''">
	<meta name="twitter:card" content="summary_large_image">	

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
    <link href="../../html/CSS/mystyle.css" rel="stylesheet"/>
    

    <style>
        p {
            text-align: justify !important;
            text-justify: inter-word !important;
        }
    </style>
    
    
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="description" content="">
    <meta name="author" content="">

    <title>Clean Blog - Sample Post</title>

    <!-- Bootstrap Core CSS -->
    <link href="../../vendor/bootstrap/css/bootstrap.min.css" rel="stylesheet">

    <!-- Theme CSS -->
    <link href="../../html/CSS/clean-blog.css" rel="stylesheet">

    <!-- Custom Fonts -->
    <link href="../../vendor/font-awesome/css/font-awesome.min.css" rel="stylesheet" type="text/css">
    <link href='https://fonts.googleapis.com/css?family=Lora:400,700,400italic,700italic' rel='stylesheet' type='text/css'>
    <link href='https://fonts.googleapis.com/css?family=Open+Sans:300italic,400italic,600italic,700italic,800italic,400,300,600,700,800' rel='stylesheet' type='text/css'>
    '''

# parse script as BeautifulSoup object
html_3 = BeautifulSoup(script_3, 'html.parser')

# replace the old font with the new font
soup.head.find(text=re.compile(r'HTML-CSS')).parent.replace_with(html_3);


# you have to render soup again (for some reason) before you can search it
soup = BeautifulSoup(soup.renderContents(),"lxml")

# remove old title
soup.body.find_all('h1')[0].decompose()


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
