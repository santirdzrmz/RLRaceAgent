using UnityEngine;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;

public class RaceAgentG06 : RaceAgentBase
{
    // ============================================================================
    // Reward shaping constants. Kept as private const so they DO NOT appear in the
    // inspector (the assignment forbids modifying anything on the Race Agent
    // component other than Max Step and the existing Settings fields).
    // ============================================================================
    private const float REWARD_CHECKPOINT = 1.0f;
    private const float REWARD_FINISH_LINE = 2.0f;
    private const float REWARD_FORWARD_SPEED = 0.001f;   // per-step, scaled by forward speed
    private const float PENALTY_WALL_HIT = -1.0f;
    private const float PENALTY_PER_STEP = -0.0005f; // small time penalty
    private const float PENALTY_BACKWARDS = -0.001f;  // per-step if moving backwards
    private const float IDLE_SPEED_THRESHOLD = 0.3f;     // |forward speed| under this counts as idle
    private const int IDLE_STEPS_TO_END = 200;      // ~10s at decision period 5
    private const float DIST_NORMALIZATION = 100f;     // for scaling distance-to-checkpoint observation

    // ============================================================================
    // Internal state (not serialized, not exposed in inspector)
    // ============================================================================
    private Vector3[] targetPositions;   // checkpoints + finish lines, cached at Start()
    private int idleStepCounter = 0;

    void Start()
    {
        // *********** DO NOT TOUCH THIS (necessary to run the drone) ***************
        ApplyMaterials();

        if (debugMode)
        {
            raySensor = GetComponent<RayPerceptionSensorComponent3D>();

            // How many rays we have in the Ray Perception Sensor 3D
            NumOfRayOutputsSensor = RayPerceptionSensor
                .Perceive(raySensor.GetRayPerceptionInput(), false)
                .RayOutputs
                .Length;

            InitializeRaysRenderers();
            ActivateRaysVisualization(true);
        }

        rb = GetComponent<Rigidbody>();
        raceManager = FindFirstObjectByType<RaceManager>();

        stopDrone = false;
        checkpoint = false;
        // **************************************************************************

        // Cache target positions for the "direction to next target" observation.
        // Done once at Start() so we don't pay the FindGameObjectsWithTag cost
        // every step.
        CacheTargetPositions();
    }

    private void CacheTargetPositions()
    {
        var checkpoints = GameObject.FindGameObjectsWithTag("checkpoint");
        var finishLines = GameObject.FindGameObjectsWithTag("finishLine");
        var combined = new List<Vector3>(checkpoints.Length + finishLines.Length);
        foreach (var c in checkpoints) combined.Add(c.transform.position);
        foreach (var f in finishLines) combined.Add(f.transform.position);
        targetPositions = combined.ToArray();
    }

    public override void OnEpisodeBegin()
    {
        // Reset our shaping state. Must come before MaxStep reset so it's clean.
        idleStepCounter = 0;

        // *********** DO NOT TOUCH THIS (necessary to run the drone) ***************
        currentStep = 0;

        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        transform.position = startingLoc.position;
        transform.rotation = startingLoc.rotation;
        // **************************************************************************
    }

    // ============================================================================
    // OBSERVATIONS
    //   Vector observation size = 10:
    //     - local linear velocity   (3)
    //     - local angular velocity  (3)
    //     - local direction to nearest checkpoint/finish (3)
    //     - normalized distance to nearest target        (1)
    //   The Ray Perception Sensor 3D contributes its own observations on top
    //   of these, automatically — no need to add them here.
    // ============================================================================
    public override void CollectObservations(VectorSensor sensor)
    {
        // 1. Local-frame linear velocity (scaled to keep values roughly in [-1, 1])
        Vector3 localVel = transform.InverseTransformDirection(rb.linearVelocity);
        sensor.AddObservation(localVel / 10f);

        // 2. Local-frame angular velocity
        Vector3 localAngVel = transform.InverseTransformDirection(rb.angularVelocity);
        sensor.AddObservation(localAngVel);

        // 3. Direction (local frame) to the nearest target + normalized distance.
        //    Falls back to "forward, far away" when nothing is found.
        Vector3 dirLocal = Vector3.forward;
        float distNorm = 1f;

        if (targetPositions != null && targetPositions.Length > 0)
        {
            float bestDist = float.MaxValue;
            Vector3 bestDir = Vector3.forward;
            foreach (var p in targetPositions)
            {
                Vector3 d = p - transform.position;
                float dist = d.magnitude;
                if (dist < bestDist && dist > 0.01f)
                {
                    bestDist = dist;
                    bestDir = d / dist;
                }
            }
            dirLocal = transform.InverseTransformDirection(bestDir);
            distNorm = Mathf.Clamp01(bestDist / DIST_NORMALIZATION);
        }
        sensor.AddObservation(dirLocal);
        sensor.AddObservation(distNorm);
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("track"))
        {
            Debug.Log("Hit!");
            // Heavy penalty for hitting a wall, then reset the episode.
            AddReward(PENALTY_WALL_HIT);
            EndEpisode();
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.CompareTag("finishLine"))
        {
            // Existing flag logic (used by RaceManager) - DO NOT REMOVE.
            checkpoint = false;
            finishLine = true;
            // Reward shaping: crossing the finish line is the primary positive event.
            // We do NOT EndEpisode here — let the agent keep going so it can learn
            // multi-lap behaviour. Episode ends naturally on MaxStep or wall hit.
            AddReward(REWARD_FINISH_LINE);
            return;
        }

        if (other.gameObject.CompareTag("checkpoint"))
        {
            // Existing flag logic (used by RaceManager) - DO NOT REMOVE.
            finishLine = false;
            checkpoint = true;
            // Reward shaping: small reward for reaching a mid-track checkpoint.
            AddReward(REWARD_CHECKPOINT);
        }
    }

    public void MoveAgent(ActionSegment<float> contAct, ActionSegment<int> disAct)
    {
        // *********** DO NOT TOUCH THIS (necessary to run the drone) ***************
        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        var speed = contAct[0];
        var actionTurn = disAct[0];

        // Looking if the agent has to turn left or right
        switch (actionTurn)
        {
            case 1: // Turn right
                rotateDir = transform.up * 1f;
                break;
            case 2: // Turn left
                rotateDir = transform.up * -1f;
                break;
        }

        dirToGo = transform.forward;

        // Apply the actions to the agent
        transform.Rotate(rotateDir, Time.fixedDeltaTime * 150f);
        rb.AddForce(dirToGo * baseDroneSpeed * speed, ForceMode.VelocityChange);
        // **************************************************************************
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (raceManager != null && (!raceManager.IsRaceActive || stopDrone))
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
            return;
        }

        currentStep++;
        if (MaxStep > 0 && currentStep >= MaxStep - 1)
        {
            // Reserved hook for end-of-episode logic. Left empty intentionally.
        }

        // Move the agent using the actions predicted by the policy
        MoveAgent(actions.ContinuousActions, actions.DiscreteActions);

        // ====================== Per-step reward shaping ===========================
        // Forward speed component (encourages going FAST in the right direction).
        float forwardSpeed = Vector3.Dot(rb.linearVelocity, transform.forward);
        if (forwardSpeed > 0f)
        {
            AddReward(REWARD_FORWARD_SPEED * forwardSpeed);
        }
        else
        {
            // Moving backwards or not moving forward at all is mildly penalized.
            AddReward(PENALTY_BACKWARDS);
        }

        // Constant time penalty - encourages finishing rather than wandering.
        AddReward(PENALTY_PER_STEP);

        // Idle detection - end the episode early if the drone gets stuck.
        if (Mathf.Abs(forwardSpeed) < IDLE_SPEED_THRESHOLD)
        {
            idleStepCounter++;
            if (idleStepCounter > IDLE_STEPS_TO_END)
            {
                AddReward(-0.5f);
                EndEpisode();
            }
        }
        else
        {
            idleStepCounter = 0;
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        var continuousActionsOut = actionsOut.ContinuousActions;

        // Speed
        continuousActionsOut[0] = Input.GetAxis("Vertical");

        // Left / Right actions
        if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[0] = 1;
        }
        else if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[0] = 2;
        }
    }
}