from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
import json
import os
#import kmrec
import pandas as pd
import numpy as np
import random

# Initialize recommender object once, so that the plk model is imported just
# on server startup

# recommender = kmrec()

@csrf_exempt
def img_upload(request):
    print('removing previous files')
    for root, dirs, files in os.walk(settings.MEDIA_ROOT):
        for filename in files:
            os.remove(settings.MEDIA_ROOT + '/' + filename)

    print('saving new files')
    fs = FileSystemStorage()
    for i, img in request.FILES.items():
        print(str(i) + ': ' + img.name)
        filename = fs.save(img.name, img)
    return HttpResponse('{"success": true}')
