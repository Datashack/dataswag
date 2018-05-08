from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
import json
import os
import leSirenuse.backend
import pandas as pd
import numpy as np
import random

backend = leSirenuse.backend.Backend(settings.MEDIA_ROOT)

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
    backend.compute_pics_presence()
    return HttpResponse('{"success": true}')

@csrf_exempt
def get_ranked_pics(request):
    target = request.POST['target']
    targets = {
        'Summer_Lovers' : 0,
        'Night_People' : 1,
        'Sport_Addicts' : 2
    }
    scores = backend.get_scores(targets[target])
    rank = {}
    rank_list = []
    for index, row in scores[scores['community'] == targets[target]].iterrows():
        rank_list.append({
            'pic_url': row['picture_uploaded'],
            'score': row['KL_score']
            })
    rank['rank'] = rank_list
    return HttpResponse(json.dumps(rank))


@csrf_exempt
def get_scored_pics(request):
    targets = {
        0 : 'Summer_Lovers',
        1 : 'Night_People',
        2 : 'Sport_Addicts'
    }
    scores_json = {}
    score_list = []
    scores = backend.get_scores(0)
    for root, dirs, files in os.walk(settings.MEDIA_ROOT):
        for filename in files:
            best_match = scores[scores['picture_uploaded']==filename].sort_values('KL_score').iloc[0]['community']
            score_list.append({
                'pic_url': filename,
                'community': targets[best_match]
                })
    scores_json["scores"] = score_list
    return HttpResponse(json.dumps(scores_json))

@csrf_exempt
def get_brands_similarity(request):
    distance = backend.get_competitors_distance()
    distance_dict = {}
    distance_list = []
    row = distance[distance['File'] == request.POST['pic']]
    brands = list(distance)[:9]
    for brand in brands:
        distance_list.append({
            'name': brand,
            'sim': row[brand].values[0]
        })
    distance_dict['brands_sim'] = distance_list
    return HttpResponse(json.dumps(distance_dict))
