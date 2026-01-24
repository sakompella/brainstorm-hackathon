import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

interface FlowFieldProps {
  neuralData: number[] | null
  gridSize: number
}

const PARTICLE_COUNT = 500

export function FlowField({ neuralData, gridSize }: FlowFieldProps) {
  const pointsRef = useRef<THREE.Points>(null)
  const velocitiesRef = useRef<Float32Array>(new Float32Array(PARTICLE_COUNT * 2))

  // Initialize particle positions and velocities
  const { positions, colors } = useMemo(() => {
    const positions = new Float32Array(PARTICLE_COUNT * 3)
    const colors = new Float32Array(PARTICLE_COUNT * 3)
    const velocities = velocitiesRef.current

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      // Random positions in normalized space [-1, 1]
      positions[i * 3] = (Math.random() - 0.5) * 2
      positions[i * 3 + 1] = (Math.random() - 0.5) * 2
      positions[i * 3 + 2] = 0

      // White-ish particles
      colors[i * 3] = 0.9
      colors[i * 3 + 1] = 0.9
      colors[i * 3 + 2] = 1.0

      // Random initial velocities
      velocities[i * 2] = (Math.random() - 0.5) * 0.01
      velocities[i * 2 + 1] = (Math.random() - 0.5) * 0.01
    }

    return { positions, colors }
  }, [])

  // Compute flow field from neural data
  const getFlowVector = (x: number, y: number): [number, number] => {
    if (!neuralData || neuralData.length === 0) {
      return [0, 0]
    }

    // Convert position to grid coordinates
    const gridX = Math.floor(((x + 1) / 2) * gridSize)
    const gridY = Math.floor(((y + 1) / 2) * gridSize)

    // Clamp to valid range
    const clampedX = Math.max(0, Math.min(gridSize - 1, gridX))
    const clampedY = Math.max(0, Math.min(gridSize - 1, gridY))

    // Get surrounding values for gradient
    const idx = clampedY * gridSize + clampedX
    const value = neuralData[idx] || 0

    // Compute gradient (flow toward higher activity)
    let dx = 0, dy = 0

    if (clampedX > 0) {
      dx -= neuralData[idx - 1] || 0
    }
    if (clampedX < gridSize - 1) {
      dx += neuralData[idx + 1] || 0
    }
    if (clampedY > 0) {
      dy -= neuralData[idx - gridSize] || 0
    }
    if (clampedY < gridSize - 1) {
      dy += neuralData[idx + gridSize] || 0
    }

    // Normalize and scale
    const mag = Math.sqrt(dx * dx + dy * dy) + 0.001
    const scale = Math.min(Math.abs(value) * 50, 0.02) // Scale by local activity

    return [dx / mag * scale, dy / mag * scale]
  }

  useFrame(() => {
    if (!pointsRef.current) return

    const positions = pointsRef.current.geometry.attributes.position.array as Float32Array
    const velocities = velocitiesRef.current

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const px = positions[i * 3]
      const py = positions[i * 3 + 1]

      // Get flow vector at particle position
      const [fx, fy] = getFlowVector(px, py)

      // Update velocity with flow influence
      velocities[i * 2] = velocities[i * 2] * 0.95 + fx
      velocities[i * 2 + 1] = velocities[i * 2 + 1] * 0.95 + fy

      // Update position
      positions[i * 3] += velocities[i * 2]
      positions[i * 3 + 1] += velocities[i * 2 + 1]

      // Wrap around edges
      if (positions[i * 3] < -1) positions[i * 3] = 1
      if (positions[i * 3] > 1) positions[i * 3] = -1
      if (positions[i * 3 + 1] < -1) positions[i * 3 + 1] = 1
      if (positions[i * 3 + 1] > 1) positions[i * 3 + 1] = -1
    }

    pointsRef.current.geometry.attributes.position.needsUpdate = true
  })

  // Create buffer attributes using THREE.js directly
  const positionAttr = useMemo(() => new THREE.BufferAttribute(positions, 3), [positions])
  const colorAttr = useMemo(() => new THREE.BufferAttribute(colors, 3), [colors])

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <primitive attach="attributes-position" object={positionAttr} />
        <primitive attach="attributes-color" object={colorAttr} />
      </bufferGeometry>
      <pointsMaterial
        size={0.02}
        vertexColors
        transparent
        opacity={0.8}
        sizeAttenuation
      />
    </points>
  )
}
