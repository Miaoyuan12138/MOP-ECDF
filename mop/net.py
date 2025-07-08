from __future__ import annotations
import logging, uuid
from typing import Tuple
import zmq

from .config import DictionaryConfig

_CTX = zmq.Context.instance() 

def _inproc(tag: str) -> str:
    return f"inproc://{tag}"

def bind_pub_pull(cfg: DictionaryConfig) -> Tuple[zmq.Socket, zmq.Socket]:
    pub  = _CTX.socket(zmq.PUB)
    pull = _CTX.socket(zmq.PULL)
    if cfg.test_leader:
        pub.bind(_inproc("mop_pub"))
        pull.bind(_inproc("mop_pull"))
        logging.info("Leader inproc sockets ready")
    else:
        pub.bind(f"tcp://*:{cfg.pub_port}")
        pull.bind(f"tcp://*:{cfg.pull_port}")
        logging.info("Leader tcp sockets bound on *:%d / *:%d",
                     cfg.pub_port, cfg.pull_port)
    return pub, pull

def connect_sub_push(cfg: DictionaryConfig) -> Tuple[zmq.Socket, zmq.Socket]:
    sub  = _CTX.socket(zmq.SUB)
    push = _CTX.socket(zmq.PUSH)
    if cfg.test_leader:
        sub.connect(_inproc("mop_pub"))
        push.connect(_inproc("mop_pull"))
        logging.info("Worker inproc sockets ready")
    else:
        sub.connect(f"tcp://{cfg.leader_ip}:{cfg.pub_port}")
        push.connect(f"tcp://{cfg.leader_ip}:{cfg.pull_port}")
        logging.info("Worker tcp sockets connected to %s:%d / %d",
                     cfg.leader_ip, cfg.pub_port, cfg.pull_port)

    sub.setsockopt(zmq.SUBSCRIBE, b"")
    return sub, push
