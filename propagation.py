def propagation_cycle(sensor_list):
    for sensor in sensor_list:
        sensor.feed_forward()
