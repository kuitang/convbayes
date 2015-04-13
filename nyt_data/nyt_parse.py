import json
import re

def parse_ny_json(filename):
    json_data=open(filename)
    data=json.load(json_data)
    body=data['cms']['article']['body']
    body_notag=re.sub("<.*?>", " ", body)
    body_notag=body_notag.replace('\n','') #remove new line
    body_notag=body_notag.replace('\u','')
    body=body_notag.replace('\u201d','')
    body=body.replace('\u201c','')

    #may need to do some other checks

    return body

    
