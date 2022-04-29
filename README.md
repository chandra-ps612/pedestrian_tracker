# people_tracker

For this demo project, I used Alex Bewley’s SORT algorithm(simple online and realtime tracking) and customized it as per the requirements.
Link: https://github.com/abewley/sort
For detection, I used **yolov5m.onnx model**.

**An instruction from the official SORT algorithm Repo** How to implement it in our own projects? 

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