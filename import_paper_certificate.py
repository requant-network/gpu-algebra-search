"""Extract the attributed numerical appendix certificate, never execute its code."""
import base64,hashlib,json,pathlib,re,urllib.request,zlib
from html.parser import HTMLParser
from experiments.tensor import verify

URL='https://arxiv.org/html/2609.17533v2'
class Text(HTMLParser):
    def __init__(self):super().__init__();self.parts=[]
    def handle_data(self,data):self.parts.append(data)
raw=urllib.request.urlopen(URL,timeout=60).read();parser=Text();parser.feed(raw.decode('utf-8'))
text=' '.join(parser.parts);match=re.search(r'BEGIN-RANK245-7X7X7-ZLIB-BASE32\s+([A-Z2-7\s]+?)\s+END-RANK245-7X7X7-ZLIB-BASE32',text)
assert match,'Appendix certificate markers not found'
encoded=''.join(match.group(1).split());encoded+='='*(-len(encoded)%8)
data=zlib.decompress(base64.b32decode(encoded));assert len(data)==245*21
scheme=[[int.from_bytes(data[offset+p:offset+p+7],'little') for p in (0,7,14)] for offset in range(0,len(data),21)]
assert verify(scheme,(7,7,7))
record=dict(shape=[7,7,7],rank=245,scheme=scheme,origin=URL,
    authors=['Zeke Medley','Anagha Gokul','Quan Luu','Panagiotis Manolios'],license='CC-BY-4.0',
    downloaded_html_sha256=hashlib.sha256(raw).hexdigest(),certificate_bytes_sha256=hashlib.sha256(data).hexdigest(),
    convention='Appendix A, three row-major seven-byte little-endian factor masks per term; independently verified without executing the supplied program.')
pathlib.Path('fixtures/paper-7x7x7-rank245.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
print('Published rank-245 numerical certificate independently verified')
