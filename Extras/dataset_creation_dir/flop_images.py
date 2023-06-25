import os

folder = 'images/u'
for file in os.listdir(folder):
   filename = file.replace(".jpg", "")
   new = f"{filename}_flop.jpg"
   cmd = f'convert {folder}/{file} -flop {folder}/{new}'
   os.system(cmd)
