import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Line } from '@react-three/drei'
import * as THREE from 'three'

interface HeatGridProps {
  neuralData: number[] | null
  gridSize: number
}

// Magma-inspired colormap
const COLORMAP = [
  [0.001, 0.000, 0.014],
  [0.074, 0.052, 0.194],
  [0.198, 0.072, 0.316],
  [0.326, 0.074, 0.370],
  [0.495, 0.074, 0.370],
  [0.665, 0.074, 0.309],
  [0.829, 0.085, 0.201],
  [0.940, 0.116, 0.104],
  [0.988, 0.212, 0.022],
  [0.988, 0.450, 0.120],
  [0.988, 0.700, 0.280],
  [0.988, 0.900, 0.550],
]

function valueToColor(value: number, vMin = -0.03, vMax = 0.03): [number, number, number] {
  const normalized = Math.max(0, Math.min(1, (value - vMin) / (vMax - vMin)))
  const idx = normalized * (COLORMAP.length - 1)
  const lowIdx = Math.floor(idx)
  const highIdx = Math.min(lowIdx + 1, COLORMAP.length - 1)
  const t = idx - lowIdx

  return [
    COLORMAP[lowIdx][0] + t * (COLORMAP[highIdx][0] - COLORMAP[lowIdx][0]),
    COLORMAP[lowIdx][1] + t * (COLORMAP[highIdx][1] - COLORMAP[lowIdx][1]),
    COLORMAP[lowIdx][2] + t * (COLORMAP[highIdx][2] - COLORMAP[lowIdx][2]),
  ]
}

export function HeatGrid({ neuralData, gridSize }: HeatGridProps) {
  const pointsRef = useRef<THREE.Points>(null)
  const timeRef = useRef(0)

  const channelCount = gridSize * gridSize

  // Base positions for the grid
  const { basePositions, positions, colors, sizes } = useMemo(() => {
    const basePositions = new Float32Array(channelCount * 3)
    const positions = new Float32Array(channelCount * 3)
    const colors = new Float32Array(channelCount * 3)
    const sizes = new Float32Array(channelCount)

    for (let row = 0; row < gridSize; row++) {
      for (let col = 0; col < gridSize; col++) {
        const idx = row * gridSize + col
        const x = (col / (gridSize - 1)) * 2 - 1
        const y = (row / (gridSize - 1)) * 2 - 1

        basePositions[idx * 3] = x
        basePositions[idx * 3 + 1] = -y
        basePositions[idx * 3 + 2] = 0

        positions[idx * 3] = x
        positions[idx * 3 + 1] = -y
        positions[idx * 3 + 2] = 0

        colors[idx * 3] = 0.05
        colors[idx * 3 + 1] = 0.05
        colors[idx * 3 + 2] = 0.1

        sizes[idx] = 1.0
      }
    }

    return { basePositions, positions, colors, sizes }
  }, [gridSize, channelCount])

  const positionAttr = useMemo(() => new THREE.BufferAttribute(positions, 3), [positions])
  const colorAttr = useMemo(() => new THREE.BufferAttribute(colors, 3), [colors])
  const sizeAttr = useMemo(() => new THREE.BufferAttribute(sizes, 1), [sizes])

  useFrame((_, delta) => {
    if (!pointsRef.current) return

    timeRef.current += delta
    const t = timeRef.current

    const posArray = pointsRef.current.geometry.attributes.position.array as Float32Array
    const colorArray = pointsRef.current.geometry.attributes.color.array as Float32Array
    const sizeArray = pointsRef.current.geometry.attributes.size.array as Float32Array

    for (let i = 0; i < channelCount; i++) {
      const value = neuralData?.[i] ?? 0
      const absValue = Math.abs(value)
      const intensity = Math.min(1, absValue / 0.025)

      const [r, g, b] = valueToColor(value)
      colorArray[i * 3] = r
      colorArray[i * 3 + 1] = g
      colorArray[i * 3 + 2] = b

      const jitterAmount = intensity * 0.012
      const jitterX = Math.sin(t * 45 + i * 0.5) * jitterAmount
      const jitterY = Math.cos(t * 40 + i * 0.7) * jitterAmount

      posArray[i * 3] = basePositions[i * 3] + jitterX
      posArray[i * 3 + 1] = basePositions[i * 3 + 1] + jitterY

      const baseSizeMultiplier = 0.8 + intensity * 0.6
      const pulse = 1 + Math.sin(t * 10 + i * 0.2) * 0.15 * intensity
      sizeArray[i] = baseSizeMultiplier * pulse
    }

    pointsRef.current.geometry.attributes.position.needsUpdate = true
    pointsRef.current.geometry.attributes.color.needsUpdate = true
    pointsRef.current.geometry.attributes.size.needsUpdate = true
  })

  const material = useMemo(() => {
    return new THREE.ShaderMaterial({
      uniforms: {
        baseSize: { value: 8.0 },
      },
      vertexShader: `
        attribute float size;
        varying vec3 vColor;
        void main() {
          vColor = color;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * baseSize * (300.0 / -mvPosition.z);
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        varying vec3 vColor;
        void main() {
          vec2 center = gl_PointCoord - vec2(0.5);
          float dist = length(center);
          if (dist > 0.5) discard;
          float alpha = smoothstep(0.5, 0.2, dist);
          gl_FragColor = vec4(vColor, alpha * 0.9);
        }
      `,
      transparent: true,
      vertexColors: true,
      depthWrite: false,
    })
  }, [])

  return (
    <points ref={pointsRef} material={material}>
      <bufferGeometry>
        <primitive attach="attributes-position" object={positionAttr} />
        <primitive attach="attributes-color" object={colorAttr} />
        <primitive attach="attributes-size" object={sizeAttr} />
      </bufferGeometry>
    </points>
  )
}

// Grid overlay with border and internal lines
export function GridOverlay() {
  const extent = 1.08
  const gridLines = 33

  const { borderPoints, gridPointSets } = useMemo(() => {
    const borderPoints: [number, number, number][] = [
      [-extent, -extent, 0],
      [extent, -extent, 0],
      [extent, extent, 0],
      [-extent, extent, 0],
      [-extent, -extent, 0],
    ]

    const gridPointSets: [number, number, number][][] = []

    // Vertical lines
    for (let i = 0; i < gridLines; i++) {
      const x = (i / (gridLines - 1)) * 2 * extent - extent
      gridPointSets.push([
        [x, -extent, 0],
        [x, extent, 0],
      ])
    }

    // Horizontal lines
    for (let i = 0; i < gridLines; i++) {
      const y = (i / (gridLines - 1)) * 2 * extent - extent
      gridPointSets.push([
        [-extent, y, 0],
        [extent, y, 0],
      ])
    }

    return { borderPoints, gridPointSets }
  }, [])

  return (
    <group position={[0, 0, -0.05]}>
      {/* Outer border - brighter */}
      <Line
        points={borderPoints}
        color="#778899"
        lineWidth={1.5}
        dashed
        dashSize={0.03}
        gapSize={0.02}
      />
      {/* Internal grid lines */}
      {gridPointSets.map((points, i) => (
        <Line
          key={i}
          points={points}
          color="#556677"
          lineWidth={1}
          dashed
          dashSize={0.02}
          gapSize={0.02}
        />
      ))}
    </group>
  )
}
