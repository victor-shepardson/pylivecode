uniform vec2 size;
uniform sampler2D color;

out vec4 fragColor;

const float pi = 3.14159265359;

void main() {
  vec2 px = gl_FragCoord.xy;
  vec2 p = px/size;

  fragColor = texture(color, p);
}
