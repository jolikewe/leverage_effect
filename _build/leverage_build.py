import re
t=open('_build/leverage_template.html').read()
d=open('_build/leverage_data.json').read()
open('./index.html','w').write(t.replace('/*DATA*/',d))
