import { Canvas } from '@react-three/fiber'
import { HeatGrid, GridOverlay } from './HeatGrid'

interface SceneProps {
  neuralData: number[] | null
  gridSize: number
}

export function Scene({ neuralData, gridSize }: SceneProps) {
  return (
    <Canvas
      camera={{ position: [0, 0, 2], fov: 60 }}
      style={{ background: '#0a0a0f' }}
    >
      <GridOverlay />
      <HeatGrid neuralData={neuralData} gridSize={gridSize} />
    </Canvas>
  )
}
