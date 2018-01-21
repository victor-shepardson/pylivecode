uniform vec2 size;
uniform sampler2D color;

const float pi = 3.14159265359;

void main() {
  vec2 px = gl_FragCoord.xy;
  vec2 p = px/size;

  vec4 c = texture2D(color, p);

  gl_FragColor = c;
}
