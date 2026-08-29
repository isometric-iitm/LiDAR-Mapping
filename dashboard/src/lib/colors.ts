export const CLASSES = [
  { id: 0, label: "Drivable", color: "#22c55e" },
  { id: 1, label: "Terrain / Non-drivable", color: "#a16207" },
  { id: 2, label: "Static Obstacle", color: "#ef4444" },
  { id: 3, label: "Dynamic Object", color: "#3b82f6" },
] as const;

export const CLASS_COLOR = new Map<number, string>(CLASSES.map((c) => [c.id, c.color]));