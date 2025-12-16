import { vec3, mat4, quat } from 'wgpu-matrix';
import { Camera } from './camera';

type ControlMode = 'orbit' | 'fps';

export class CameraControl {
  element: HTMLCanvasElement;
  private mode: ControlMode = 'orbit';
  
  private keys = {
    w: false,
    a: false,
    s: false,
    d: false,
    q: false,
    e: false,
    shift: false,
  };
  
  private rotating = false;
  private panning = false;
  private lastX: number;
  private lastY: number;
  
  private moveSpeed = 0.1;
  private fastMoveSpeed = 0.3;
  
  private target: Float32Array;
  private distance: number;
  
  constructor(private camera: Camera) {
    this.target = vec3.create(0, 0, 0);
    this.distance = vec3.len(this.camera.position);
    
    this.register_element(camera.canvas);
    this.setupKeyboardControls();
    this.startUpdateLoop();
  }
  
  setMode(mode: ControlMode) {
    this.mode = mode;
    console.log(`Camera control mode: ${mode}`);
    
    if (mode === 'orbit') {
      document.exitPointerLock();
      this.updateTarget();
    }
  }
  
  updateTarget() {
    const forward = vec3.mulScalar(this.camera.look, this.distance);
    this.target = vec3.add(this.camera.position, forward);
  }

  register_element(value: HTMLCanvasElement) {
    if (this.element && this.element != value) {
      this.element.removeEventListener('pointerdown', this.downCallback.bind(this));
      this.element.removeEventListener('pointermove', this.moveCallback.bind(this));
      this.element.removeEventListener('pointerup', this.upCallback.bind(this));
      this.element.removeEventListener('wheel', this.wheelCallback.bind(this));
    }

    this.element = value;
    this.element.addEventListener('pointerdown', this.downCallback.bind(this));
    this.element.addEventListener('pointermove', this.moveCallback.bind(this));
    this.element.addEventListener('pointerup', this.upCallback.bind(this));
    this.element.addEventListener('wheel', this.wheelCallback.bind(this));
    this.element.addEventListener('contextmenu', (e) => { e.preventDefault(); });
  }
  
  setupKeyboardControls() {
    window.addEventListener('keydown', (e) => {
      if (this.mode !== 'fps') return;
      
      const key = e.key.toLowerCase();
      if (key === 'w') this.keys.w = true;
      if (key === 'a') this.keys.a = true;
      if (key === 's') this.keys.s = true;
      if (key === 'd') this.keys.d = true;
      if (key === 'q') this.keys.q = true;
      if (key === 'e') this.keys.e = true;
      if (key === 'shift') this.keys.shift = true;
    });
    
    window.addEventListener('keyup', (e) => {
      if (this.mode !== 'fps') return;
      
      const key = e.key.toLowerCase();
      if (key === 'w') this.keys.w = false;
      if (key === 'a') this.keys.a = false;
      if (key === 's') this.keys.s = false;
      if (key === 'd') this.keys.d = false;
      if (key === 'q') this.keys.q = false;
      if (key === 'e') this.keys.e = false;
      if (key === 'shift') this.keys.shift = false;
    });
  }
  
  startUpdateLoop() {
    const update = () => {
      if (this.mode === 'fps') {
        this.updateMovement();
      }
      requestAnimationFrame(update);
    };
    requestAnimationFrame(update);
  }
  
  updateMovement() {
    if (!Object.values(this.keys).some(v => v)) {
      return;
    }
    
    const speed = this.keys.shift ? this.fastMoveSpeed : this.moveSpeed;
    let moved = false;
    
    if (this.keys.w) {
      const delta = vec3.mulScalar(this.camera.look, speed);
      vec3.add(this.camera.position, delta, this.camera.position);
      moved = true;
    }
    if (this.keys.s) {
      const delta = vec3.mulScalar(this.camera.look, -speed);
      vec3.add(this.camera.position, delta, this.camera.position);
      moved = true;
    }
    if (this.keys.a) {
      const delta = vec3.mulScalar(this.camera.right, -speed);
      vec3.add(this.camera.position, delta, this.camera.position);
      moved = true;
    }
    if (this.keys.d) {
      const delta = vec3.mulScalar(this.camera.right, speed);
      vec3.add(this.camera.position, delta, this.camera.position);
      moved = true;
    }
    if (this.keys.q) {
      const delta = vec3.mulScalar(this.camera.up, -speed);
      vec3.add(this.camera.position, delta, this.camera.position);
      moved = true;
    }
    if (this.keys.e) {
      const delta = vec3.mulScalar(this.camera.up, speed);
      vec3.add(this.camera.position, delta, this.camera.position);
      moved = true;
    }
    
    if (moved) {
      this.camera.update_buffer();
    }
  }

downCallback(event: PointerEvent) {
  if (!event.isPrimary) {
    return;
  }
  
  if (this.mode === 'orbit') {
    if (event.button === 2) {
      if (event.shiftKey) {
        this.panning = true;
        this.rotating = false;
      } else {
        this.rotating = true;
        this.panning = false;
      }
    } else if (event.button === 1) {
      this.panning = true;
      this.rotating = false;
    } else if (event.button === 0 && event.shiftKey) {
      this.panning = true;
      this.rotating = false;
    } else {
      return;
    }
    this.lastX = event.pageX;
    this.lastY = event.pageY;
  } else {
    // FPS mode
    if (event.button === 2 && event.shiftKey) {
      this.panning = true;
      this.rotating = false;
    } else if (event.button === 1) {
      this.panning = true;
      this.rotating = false;
    } else if (event.button === 0 && event.shiftKey) {
      this.panning = true;
      this.rotating = false;
    } else if (event.button === 2) {
      this.rotating = true;
      this.panning = false;
      this.element.requestPointerLock();
    } else {
      return;
    }
    this.lastX = event.pageX;
    this.lastY = event.pageY;
  }
}

moveCallback(event: PointerEvent) {
  if (!(this.rotating || this.panning)) {
    return;
  }

  const xDelta = this.mode === 'fps' && this.rotating 
    ? (event.movementX || (event.pageX - this.lastX))
    : (event.pageX - this.lastX);
  const yDelta = this.mode === 'fps' && this.rotating
    ? (event.movementY || (event.pageY - this.lastY))
    : (event.pageY - this.lastY);
  
  this.lastX = event.pageX;
  this.lastY = event.pageY;

  if (this.mode === 'orbit') {
    if (this.rotating) {
      this.rotateAroundTarget(xDelta, yDelta);
    } else if (this.panning) {
      this.panRhino(xDelta, yDelta);
    }
  } else {
    if (this.rotating) {
      this.rotateFPS(xDelta, yDelta);
    } else if (this.panning) {
      this.panFPS(xDelta, yDelta);
    }
  }
}
  
  upCallback(event: PointerEvent) {
    this.rotating = false;
    this.panning = false;
    
    if (this.mode === 'fps') {
      document.exitPointerLock();
    }
    
    event.preventDefault();
  }
  
  wheelCallback(event: WheelEvent) {
    event.preventDefault();
    
    if (this.mode === 'orbit') {
      const zoomSpeed = 0.001;
      this.distance *= (1.0 + event.deltaY * zoomSpeed);
      this.distance = Math.max(0.1, Math.min(this.distance, 1000.0));
      
      const direction = vec3.sub(this.camera.position, this.target);
      vec3.normalize(direction, direction);
      vec3.mulScalar(direction, this.distance, direction);
      vec3.add(this.target, direction, this.camera.position);
      
      this.camera.update_buffer();
    } else {
      const delta = vec3.mulScalar(this.camera.look, -event.deltaY * 0.001);
      vec3.add(delta, this.camera.position, this.camera.position);
      this.camera.update_buffer();
    }
  }

rotateAroundTarget(xDelta: number, yDelta: number) {
  const sensitivity = 0.005;
  
  const toCamera = vec3.sub(this.camera.position, this.target);
  
  const right = vec3.cross(toCamera, this.camera.up);
  vec3.normalize(right, right);
  
  const horizontalRot = mat4.rotationY(-xDelta * sensitivity);
  vec3.transformMat4(toCamera, horizontalRot, toCamera);
  
  const verticalRot = mat4.fromQuat(quat.fromAxisAngle(right, -yDelta * sensitivity));
  vec3.transformMat4(toCamera, verticalRot, toCamera);
  
  vec3.add(this.target, toCamera, this.camera.position);
  
  const newLook = vec3.normalize(vec3.negate(toCamera));
  const newRight = vec3.normalize(vec3.cross(newLook, [0, 1, 0]));
  const newUp = vec3.cross(newRight, newLook);
  
  this.camera.rotation = mat4.fromMat3([
    ...newRight, 0,
    ...newUp, 0,
    ...vec3.negate(newLook), 0,
    0, 0, 0, 1
  ]);
  
  this.camera.update_buffer();
}
  rotateFPS(xDelta: number, yDelta: number) {
    const sensitivity = 0.002;
    const yaw = -xDelta * sensitivity;
    const pitch = -yDelta * sensitivity;
    
    const right = this.camera.right;
    const up = vec3.fromValues(0, 1, 0);
    
    const yawRot = mat4.fromQuat(quat.fromAxisAngle(up, yaw));
    const pitchRot = mat4.fromQuat(quat.fromAxisAngle(right, pitch));
    
    const rotation = mat4.mul(yawRot, pitchRot);
    mat4.mul(rotation, this.camera.rotation, this.camera.rotation);
    
    this.camera.update_buffer();
  }

  panRhino(xDelta: number, yDelta: number) {
    const panSpeed = this.distance * 0.001;
    
    const right = vec3.mulScalar(this.camera.right, -xDelta * panSpeed);
    const up = vec3.mulScalar(this.camera.up, -yDelta * panSpeed);
    
    vec3.add(this.camera.position, right, this.camera.position);
    vec3.add(this.camera.position, up, this.camera.position);
    vec3.add(this.target, right, this.target);
    vec3.add(this.target, up, this.target);
    
    this.camera.update_buffer();
  }



  panFPS(xDelta: number, yDelta: number) {
  const panSpeed = 0.01;
  
  const right = vec3.mulScalar(this.camera.right, -xDelta * panSpeed);
  const up = vec3.mulScalar(this.camera.up, -yDelta * panSpeed);
  
  vec3.add(this.camera.position, right, this.camera.position);
  vec3.add(this.camera.position, up, this.camera.position);
  
  this.camera.update_buffer();
}
};

