(define (problem task)
(:domain robot_delivery)
(:objects
wp0 wp1 wp2 wp3 wp4 wp5 - waypoint
machine0 machine1 machine2 - machine
robot0 robot1 robot2 - robot
)
(:init
    (robot_at robot0 wp0)
    (nocarrying_order robot0)
    (undocked robot0)
    (localised robot0)

    (robot_at robot1 wp0)
    (nocarrying_order robot1)
    (undocked robot1)
    (localised robot1)

    (robot_at robot2 wp0)
    (nocarrying_order robot2)
    (undocked robot2)
    (localised robot2)

    (machine_off machine0)
    (machine_off machine1)
    (machine_off machine2)

    (delivery_destination wp5)
    (delivery_destination wp2)

    (= (distance wp0 wp1) 9)
    (= (distance wp1 wp0) 9)
    (= (distance wp0 wp2) 11)
    (= (distance wp2 wp0) 11)
    (= (distance wp0 wp3) 4)
    (= (distance wp3 wp0) 4)
    (= (distance wp0 wp4) 8)
    (= (distance wp4 wp0) 8)
    (= (distance wp0 wp5) 14)
    (= (distance wp5 wp0) 14)
    (= (distance wp1 wp2) 19)
    (= (distance wp2 wp1) 19)
    (= (distance wp1 wp3) 12)
    (= (distance wp3 wp1) 12)
    (= (distance wp1 wp4) 2)
    (= (distance wp4 wp1) 2)
    (= (distance wp1 wp5) 10)
    (= (distance wp5 wp1) 10)
    (= (distance wp2 wp3) 8)
    (= (distance wp3 wp2) 8)
    (= (distance wp2 wp4) 18)
    (= (distance wp4 wp2) 18)
    (= (distance wp2 wp5) 16)
    (= (distance wp5 wp2) 16)
    (= (distance wp3 wp4) 11)
    (= (distance wp4 wp3) 11)
    (= (distance wp3 wp5) 11)
    (= (distance wp5 wp3) 11)
    (= (distance wp4 wp5) 9)
    (= (distance wp5 wp4) 9)
    (= (distance wp0 machine0) 7)
    (= (distance machine0 wp0) 7)
    (= (distance wp0 machine1) 5)
    (= (distance machine1 wp0) 5)
    (= (distance wp0 machine2) 21)
    (= (distance machine2 wp0) 21)
    (= (distance wp1 machine0) 15)
    (= (distance machine0 wp1) 15)
    (= (distance wp1 machine1) 5)
    (= (distance machine1 wp1) 5)
    (= (distance wp1 machine2) 17)
    (= (distance machine2 wp1) 17)
    (= (distance wp2 machine0) 7)
    (= (distance machine0 wp2) 7)
    (= (distance wp2 machine1) 15)
    (= (distance machine1 wp2) 15)
    (= (distance wp2 machine2) 21)
    (= (distance machine2 wp2) 21)
    (= (distance wp3 machine0) 10)
    (= (distance machine0 wp3) 10)
    (= (distance wp3 machine1) 8)
    (= (distance machine1 wp3) 8)
    (= (distance wp3 machine2) 18)
    (= (distance machine2 wp3) 18)
    (= (distance wp4 machine0) 14)
    (= (distance machine0 wp4) 14)
    (= (distance wp4 machine1) 4)
    (= (distance machine1 wp4) 4)
    (= (distance wp4 machine2) 16)
    (= (distance machine2 wp4) 16)
    (= (distance wp5 machine0) 20)
    (= (distance machine0 wp5) 20)
    (= (distance wp5 machine1) 12)
    (= (distance machine1 wp5) 12)
    (= (distance wp5 machine2) 8)
    (= (distance machine2 wp5) 8)
    (= (distance machine0 machine1) 11)
    (= (distance machine1 machine0) 11)
    (= (distance machine0 machine2) 27)
    (= (distance machine2 machine0) 27)
    (= (distance machine1 machine2) 19)
    (= (distance machine2 machine1) 19)
)
(:goal (and
    (order_delivered wp5)
    (order_delivered wp2)
))
)
