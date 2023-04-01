import pika
import json
from datetime import datetime


class PassengerCountMessage:
    def __init__(self, id_detection: int,
                 predicted_class: int,
                 class_name: str,
                 direction: int,
                 train_number: str,
                 confidence_score: float,
                 latitude: float = 48.224490,
                 longitude: float = 16.474500):
        self.id_detection = id_detection
        self.predicted_class = predicted_class
        self.class_name = class_name
        self.direction = direction
        self.train_number = train_number
        self.confidence_score = confidence_score
        self.time_utc = str(datetime.utcnow())
        self.latitude = latitude
        self.longitude = longitude

    def to_json(self):
        return json.dumps(self, default=lambda message: message.__dict__, sort_keys=True, indent=4)


class MessageQueue:
    def __init__(self, queue_name: str = "passenger_count", ip_address="localhost"):
        self.queue_name = queue_name
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(ip_address))
        self.channel = self.connection.channel()

        self.channel.queue_declare(queue=self.queue_name)

    def publish(self, id_detection: int, predicted_class: int, class_name: str, direction: int, train_number: str, confidence_score: float):
        dto = PassengerCountMessage(id_detection, predicted_class, class_name, direction, train_number, confidence_score)
        self.channel.basic_publish(exchange='', routing_key=self.queue_name, body=dto.to_json())

    def __del__(self):
        print("Closing message queue")
        self.connection.close()
