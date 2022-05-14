# people_tracker

For this demo project, I used ```yolov5m.onnx```, ```OpenCV```, and customized ```sort.py``` from ```Alex Bewley’s SORT algorithm(simple online and real-time tracking)```.


Link: https://github.com/abewley/sort

I used a simple video for tracking (partial implementation) but if you use a **complex video (When CCTV is closer and there is so much occlusion throughout the video)**, customized 
```sort.py``` might throw errors and entire frames of the video won't be processed.

You can also implement this ```repo``` having been modified ```main.py``` in other projects i.e. ```Traffic_Surveillance```, etc. The screenshot provided below is produced by the same
repo after some minor modification.


![traffic](https://user-images.githubusercontent.com/89622996/168426572-a177ce1c-be44-492b-af70-17c98a040b66.png)


### An instruction from the official SORT algorithm Repo for implementing it in our own projects 

Below is the gist of how to instantiate and update SORT. See the ['__main__'](https://github.com/abewley/sort/blob/master/sort.py#L239) section of [sort.py](https://github.com/abewley/sort/blob/master/sort.py#L239) for a complete example.
    
    from sort import *
    
    #create instance of SORT
    mot_tracker = Sort() 
    
    # get detections
    ...
    
    # update SORT
    track_bbs_ids = mot_tracker.update(detections)

    # track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
    ...
