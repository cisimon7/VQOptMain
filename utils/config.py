from datetime import datetime


config = dict(
    lanes_count=4,
    vehicles_count=50,
    vehicles_density=3.0,
    controlled_vehicles=1,
    initial_lane_id=None,
    ego_spacing=2.0,
    duration=50,
    speed_limit=15,
    other_vehicles_type="highway_env.vehicle.behavior.IDMVehicle",
    simulation_frequency=100,
    policy_frequency=25,
    observation=dict(
        type="Kinematics",
        vehicles_count=11,
        features=["x", "y", "vx", "vy", "heading"],
        features_range=dict(
            x=[-150, 150],
            y=[-10, 10],
            vx=[-30, 30],
            vy=[-10, 10]
        ),
        absolute=True,
        clip=False,
        normalize=False,
        see_behind=False
    ),
    obs_normalize=False,
    action=dict(
        type="ContinuousAction",
        longitudinal=True,
        lateral=True,
    ),
    right_lane_reward=0,
    high_speed_reward=0.4,
    collision_reward=-10,
    lane_change_reward=0,
    reward_speed_range=[20, 33],
    offroad_terminal=False,
)


def print_config_details(config, collision_rate, mean_speed, std_speed, seed, space_width=10):
    print(
            f"\nRUN DETAILS:\n"
            f"{'collision_rate':<20}  = {round(collision_rate, 2)} %\n" +
            f"{'mean_speed':<20}  = {round(float(mean_speed), 2)} \n" +
            f"{'std_speed':<20}  = {round(float(std_speed), 2)} \n" +
            f"{'vehicles_density':<20}  = {config['vehicles_density']}\n" +
            f"{'vehicles_count':<20}  = {config['vehicles_count']}\n" +
            f"{'speed_limit':<20}  = {config['speed_limit']}\n" +
            f"{'lanes_count':<20}  = {config['lanes_count']}\n" +
            f"{'seed':<20}  = {seed}\n"
    )


def print_eps_details(step: int, crashed: bool, mean_speed, duration, collision_rate):
    duration = datetime.utcfromtimestamp(duration)
    print(
        f"Episode"+f"{step}".rjust(2) +
        f", crashed="+f"{crashed}".rjust(6) +
        f", speed_mean="+f"{round(float(mean_speed), 2)}".rjust(5) +
        f", duration="+f"{duration.strftime('%H:%M:%S')}".rjust(8) +
        f", collision_rate="+f"{round(collision_rate, 2)}%".rjust(6)
    )
