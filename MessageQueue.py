import pika
import json


class PassengerCountMessage:
    def __init__(self, id_detection: int, predicted_class: int, direction: int, confidence_score: float):
        self.id_detection = id_detection
        self.predicted_class = predicted_class
        self. direction = direction
        self. confidence_score = confidence_score


class MessageQueue:
    def __init__(self, queue_name: str = "passenger_count"):
        self.queue_name = queue_name
        self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        self.channel = self.connection.channel()

        self.channel.queue_declare(queue=self.queue_name)

    def publish(self, id_detection: int, predicted_class: int, direction: int, confidence_score: float):
        dto = PassengerCountMessage(id_detection, predicted_class, direction, confidence_score)
        dto_json = json.dumps(dto)
        self.channel.basic_publish(exchange='', routing_key=self.queue_name, body=dto_json)

    # def __del__(self):
    #     print("Closing message queue")
    #     self.connection.close()
